"""
Runs the 84-record working set through OpenResearcher-30B-A3B (TIGER-AI-Lab/OpenResearcher),
in place of deploy_agent.py's benchmark-dataset loader.

Rather than editing the vendor repo to add a custom dataset to data_utils.py's fixed
split_name_map (browsecomp/hle/gaia/xbench/etc.), this script calls the same lower-level
primitives the repo's own README "Quick Start" snippet uses directly -- run_one,
BrowserPool, OpenAIAsyncGenerator -- looped over our own case list, with the same
resumability/retry/concurrency shape as deploy_agent.py's own worker_entry().

--- Setup (one-time, on the GPU box) ---
1. git clone https://github.com/TIGER-AI-Lab/OpenResearcher.git
2. cd OpenResearcher && bash setup.sh   (installs deps incl. openai_harmony, tevatron)
3. Get a Serper API key (https://serper.dev/) and set SERPER_API_KEY in .env
   (either this project's .env, or export it directly -- see --repo-dir below)
4. Start the model server: bash scripts/start_nemotron_servers.sh
   (check logs/ to confirm the vLLM server is actually up before proceeding)

--- Usage ---
Copy this file into the cloned OpenResearcher repo root (or pass --repo-dir pointing
at the clone), then:

    # Pilot first -- 5 cases, confirm round counts/timing/cost assumptions before the full run
    python run_openresearcher_batch.py --repo-dir /path/to/OpenResearcher --limit 5

    # Full batch (all 84), resumable if interrupted -- rerun the same command to continue
    python run_openresearcher_batch.py --repo-dir /path/to/OpenResearcher \\
        --output outputs/openresearcher_run/batch_results.jsonl

Launch wrapped in caffeinate + nohup for anything beyond the pilot -- this project has
been bitten twice before by unprotected long jobs going to sleep mid-run (see
PROGRESS.md's infra notes):

    nohup caffeinate -i python3 run_openresearcher_batch.py --repo-dir ... \\
        > outputs/openresearcher_run/run.log 2>&1 & disown
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import traceback

MAX_RETRY = 5


def load_cases(working_set_csv, clarification_csv, case_ids=None, limit=None):
    with open(working_set_csv) as f:
        working = {r["case_id"]: r for r in csv.DictReader(f)}
    with open(clarification_csv) as f:
        clar = {r["case_id"]: r for r in csv.DictReader(f)}

    missing = set(working) ^ set(clar)
    if missing:
        raise ValueError(f"working_set and clarification_classification case_id mismatch: {missing}")

    cases = []
    for cid, row in working.items():
        cases.append({
            "qid": cid,
            "case_id": cid,
            "source": row["source"],
            # Turn 1 only, matching the real "user's first message to a research agent"
            # scenario -- turn2 in the original chat logs was directed at a different
            # chatbot during data collection, not part of the task given to OpenResearcher.
            # If a specific case's turn2 is judged necessary context, curate it in here
            # deliberately (per this project's "self-contained task" design principle)
            # rather than blanket-including it.
            "question": row["prompt"],
            "clarification_label": clar[cid]["clarification_label"],
        })

    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c["case_id"] in wanted]
    if limit:
        cases = cases[:limit]
    return cases


def load_already_done(output_path):
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add(rec["qid"])
            except Exception:
                continue
    return done


async def process_case(case, generator, browser_pool, max_rounds, sem):
    async with sem:
        qid = case["qid"]
        attempt = 0
        error_msg = None
        t0 = time.time()
        while attempt < MAX_RETRY:
            attempt += 1
            try:
                messages = await run_one(
                    question=case["question"],
                    qid=qid,
                    generator=generator,
                    browser_pool=browser_pool,
                    max_rounds=max_rounds,
                )
                dt = time.time() - t0
                rec = dict(case)
                rec.update({
                    "messages": messages, "latency_s": dt, "error": None,
                    "attempts": attempt, "status": "success",
                })
                return rec
            except Exception:
                error_msg = traceback.format_exc()
                print(f"[qid {qid}] attempt {attempt}/{MAX_RETRY} failed:\n{error_msg}")
        rec = dict(case)
        rec.update({"messages": [], "latency_s": time.time() - t0, "error": error_msg,
                     "attempts": attempt, "status": "fail"})
        return rec


async def run_batch(args):
    cases = load_cases(args.working_set_csv, args.clarification_csv,
                        case_ids=args.case_ids, limit=args.limit)
    done = load_already_done(args.output)
    todo = [c for c in cases if c["qid"] not in done]

    print(f"Total cases selected: {len(cases)} | already done: {len(done)} | remaining: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    generator = OpenAIAsyncGenerator(
        base_url=args.vllm_base_url,
        model_name=args.model_name,
        use_native_tools=True,
    )
    browser_pool = BrowserPool(search_url=None, browser_backend=args.browser_backend)
    sem = asyncio.Semaphore(args.concurrency)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_header_note = not os.path.exists(args.output)

    tasks = [asyncio.create_task(process_case(c, generator, browser_pool, args.max_rounds, sem))
             for c in todo]

    t0 = time.time()
    n_done = 0
    n_success = 0
    n_fail = 0
    with open(args.output, "a", encoding="utf-8") as f:
        for fut in asyncio.as_completed(tasks):
            rec = await fut
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            browser_pool.cleanup(rec["qid"])
            n_done += 1
            if rec["status"] == "success":
                n_success += 1
            else:
                n_fail += 1
            elapsed = time.time() - t0
            rate = n_done / elapsed if elapsed > 0 else 0
            remaining = (len(todo) - n_done) / rate if rate > 0 else float("inf")
            print(f"  [{n_done}/{len(todo)}] qid={rec['qid']} status={rec['status']} "
                  f"latency={rec['latency_s']:.0f}s | {rate:.3f}/s, ~{remaining/60:.1f} min remaining")
            sys.stdout.flush()

    print(f"\nDone. {n_success} succeeded, {n_fail} failed after {MAX_RETRY} retries each. "
          f"Wrote results to {args.output}")
    print("Rerunning the same command skips everything already written here (success or fail), "
          "matching deploy_agent.py's own resumability -- an interrupted run picks back up where "
          "it left off, but a case that exhausted all retries needs manual retry: remove its line "
          "from the output file, or rerun with --case-ids <those qids> pointed at a fresh output path.")
    if n_fail:
        print(f"{n_fail} case(s) need one of those two.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", default=".",
                         help="Path to the cloned TIGER-AI-Lab/OpenResearcher repo "
                              "(needed to import deploy_agent / utils.openai_generator)")
    parser.add_argument("--working-set-csv", default="outputs/working_set_151sample.csv")
    parser.add_argument("--clarification-csv", default="outputs/clarification_classification.csv")
    parser.add_argument("--output", default="outputs/openresearcher_run/batch_results.jsonl")
    parser.add_argument("--vllm-base-url", default="http://localhost:8001/v1")
    parser.add_argument("--browser-backend", default="serper", choices=["serper", "free", "local"],
                         help="'serper' (paid, Google Search API + scrape), 'free' (self-hosted "
                              "SearXNG + Jina Reader, see SEARCH_BACKEND_DESIGN.md), or 'local' "
                              "(fixed BrowseComp-Plus corpus, not live web -- not useful here).")
    parser.add_argument("--model-name", default="OpenResearcher/OpenResearcher-30B-A3B")
    parser.add_argument("--max-rounds", type=int, default=50,
                         help="Ceiling per case. The repo's own benchmark runs use 200 "
                              "(sized for adversarial fact-finding questions); 50 matches "
                              "the public demo's default and is a more conservative starting "
                              "point for real, non-adversarial research-planning queries.")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only run the first N cases -- use for a pilot before the full 84.")
    parser.add_argument("--case-ids", nargs="*", default=None,
                         help="Run only these specific case_ids -- use for a targeted pilot "
                              "(e.g. a couple of the 3 NEEDS_CLARIFICATION cases plus a couple "
                              "of long/short ANSWERABLE ones).")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo_dir))
    global run_one, BrowserPool, OpenAIAsyncGenerator
    try:
        from deploy_agent import run_one, BrowserPool
        from utils.openai_generator import OpenAIAsyncGenerator
    except ImportError as e:
        print(f"Could not import from --repo-dir={args.repo_dir}: {e}")
        print("Make sure this points at a cloned TIGER-AI-Lab/OpenResearcher checkout "
              "with dependencies installed (bash setup.sh).")
        sys.exit(1)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not os.getenv("SERPER_API_KEY"):
        print("SERPER_API_KEY is not set (checked environment and .env). "
              "SerperServiceBrowserBackend will fail on the first search call.")
        sys.exit(1)

    asyncio.run(run_batch(args))


if __name__ == "__main__":
    main()

"""Run one local MLX judge over judge packets. Runs on the M5 Ultra.

Usage:
  python3 -I run_local_judge.py --judge A --cases E9 E20 F22 F10 5 F39
  python3 -I run_local_judge.py --judge B --cases ...
  python3 -I run_local_judge.py --judge B --cases E9 --packets packets_v3 --out judge_B_v3_s1 --seed 1
  python3 -I run_local_judge.py --judge A --smoke        (tiny test of the chat template and memory)

Judge A: Qwen2.5-72B-Instruct (4-bit), greedy decoding, system prompt + user message.
Judge B: Muse Glimmer 30B (8-bit), the vendor's sampling (temperature 1.0, top_p 0.95, top_k 64), system prompt
         + user message. The chat template adds "Reasoning strength: high." to the system message. It reasons first
         (a separate assistant segment); the reasoning stays in the raw file and the last JSON object is parsed.
Both see exactly the same text. Reads outputs/judge_runs/packets/<case>.md and key/<case>.json.
Writes outputs/judge_runs/judge_<A|B>/<case>.json and raw/judge_<A|B>/<case>.attemptN.txt.
"""
import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

RUNS = Path(__file__).resolve().parents[1] / "outputs" / "judge_runs"
JUDGES = {
    "A": dict(model="mlx-community/Qwen2.5-72B-Instruct-4bit", temp=0.0, top_p=1.0, max_tokens=4096, system=True),
    # First B (DeepSeek-R1-Distill-Llama-70B-4bit) produced degenerate text on the 6-9k token packets and was dropped.
    "B": dict(model="mlx-community/Muse-Glimmer-30B-8bit", temp=1.0, top_p=0.95, top_k=64, max_tokens=20000, system=True),
}


def split_packet(text):
    head, case = text.split("\n\n# CASE ", 1)
    system = head.replace("# JUDGE INSTRUCTIONS", "", 1).strip()
    return system, "# CASE " + case.strip()


def last_json(text):
    """Return the last complete JSON object in text that has an 'items' list, or None."""
    dec = json.JSONDecoder()
    found = None
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            found = obj
    return found


def validate(obj, case, ids):
    if obj is None:
        return "no JSON object with an 'items' list was found"
    got = [i.get("item_id") for i in obj["items"]]
    if got != ids:
        return f"item_id list must be exactly {ids} in that order, but it was {got}"
    for i in obj["items"]:
        if i.get("judge_verdict") not in ("pass", "fail"):
            return f"item {i.get('item_id')}: judge_verdict must be 'pass' or 'fail'"
    if (obj.get("accuracy") or {}).get("status") not in ("clean", "minor", "issues"):
        return "accuracy.status must be 'clean', 'minor' or 'issues'"
    return None


def run(judge, cases, smoke, packets="packets", out=None, seed=0, keys="key"):
    out = out or f"judge_{judge}"
    cfg = JUDGES[judge]
    print(f"loading {cfg['model']} ...", flush=True)
    t0 = time.time()
    model, tok = load(cfg["model"])
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    sampler = make_sampler(temp=cfg["temp"], top_p=cfg["top_p"], top_k=cfg.get("top_k", 0))

    def chat(messages, max_tokens):
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        n = len(tok.encode(prompt))
        t = time.time()
        out = generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)
        return out, n, len(tok.encode(out)), time.time() - t, prompt[-60:]

    if smoke:
        out, n, m, s, tail = chat([{"role": "user", "content": 'Reply with exactly this JSON and nothing else: {"ok": true}'}], 1500)
        print(f"prompt tokens {n}, output tokens {m}, {s:.1f}s, peak memory {mx.get_peak_memory() / 1e9:.1f} GB")
        print("prompt ends with:", repr(tail)); print("output:", out[:600]); return

    (RUNS / out).mkdir(parents=True, exist_ok=True)
    (RUNS / "raw" / out).mkdir(parents=True, exist_ok=True)
    for q in cases:
        mx.random.seed(seed)
        system, user = split_packet((RUNS / packets / f"{q}.md").read_text())
        ids = [i["item_id"] for i in json.loads((RUNS / keys / f"{q}.json").read_text())["items"]]
        base = [{"role": "system", "content": system}, {"role": "user", "content": user}] if cfg["system"] \
            else [{"role": "user", "content": system + "\n\n" + user}]
        messages, result, err = list(base), None, None
        for attempt in range(1, 4):
            raw, n_in, n_out, secs, _ = chat(messages, cfg["max_tokens"])
            (RUNS / "raw" / out / f"{q}.attempt{attempt}.txt").write_text(raw)
            answer = raw.split("</think>")[-1] if "</think>" in raw else raw
            obj = last_json(answer)
            err = validate(obj, q, ids)
            print(f"{q} attempt {attempt}: in {n_in} tok, out {n_out} tok, {secs:.0f}s, peak {mx.get_peak_memory() / 1e9:.0f} GB, "
                  f"{'OK' if err is None else 'INVALID: ' + err}", flush=True)
            if err is None:
                obj["meta"] = dict(model=cfg["model"], temp=cfg["temp"], top_p=cfg["top_p"], top_k=cfg.get("top_k", 0), attempts=attempt, seed=seed, packets=packets,
                                   prompt_tokens=n_in, output_tokens=n_out, seconds=round(secs))
                result = obj
                break
            messages = base + [{"role": "assistant", "content": answer.strip()[:6000]},
                               {"role": "user", "content": f"That output was not usable: {err}. Return only the corrected JSON object."}]
        if result is None:
            result = {"case_id": q, "error": err, "items": []}
        (RUNS / out / f"{q}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=["A", "B"])
    ap.add_argument("--cases", nargs="*", default=[])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--packets", default="packets")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keys", default="key")
    a = ap.parse_args()
    run(a.judge, a.cases, a.smoke, a.packets, a.out, a.seed, a.keys)

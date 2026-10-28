# Live-traffic evals: code

Code for a case study that builds an evaluation seed set from real chat traffic, runs it through an open
deep-research agent, and scores the answers against per-question checklists with a locally hosted LLM judge.

> Chutani, S. *Evaluating a Deep-Research Agent on Real-User Questions: An Exploratory Case Study in
> Measurement Sensitivity.* Preprint, 2026.

This repository holds the pipeline code and the judge prompts. It does **not** hold the data. The study does not
measure agent performance, and nothing here should be read as a benchmark of the agent.

## What is released, and what is not

| Released | Not released |
|---|---|
| Selection, audit-support and overlap-check scripts | The 84 questions, and any prompt or conversation text |
| Packet builders, judge runner, scoring and analysis scripts | Checklists, tags and split-item wording (they are derived from user conversations) |
| Judge system prompts (v2 and v3) | The agent's answers and trajectories |
| Docker files and requirements | Reference ratings and per-case scores |

The source questions come from LMSYS-Chat-1M (gated, used under its data agreement) and the Chatbot Arena
conversations. They cannot be redistributed. The scripts that classify conversations originally carried worked
examples taken from those datasets. Those examples are removed here and marked in each rubric, so labels you
produce will differ from the paper's until you add examples of your own.

Headline figures are in the paper.

## Layout

```
data_collection/
  sanity_check_query_volume.py        stage 1: sample prompts, embedding filter against exemplar queries
  classify_candidates_qwen.py         stage 2: binary classification of candidates (local Qwen2.5 7B via Ollama)
  classify_taxonomy_qwen.py           early taxonomy, replaced by the next file
  classify_taxonomy_v2_marchionini.py Lookup / Learn / Investigate taxonomy
  classify_skillbuilding_split.py     knowledge field vs practical skill
  classify_clarification_qwen.py      answerable vs needs clarification
  topic_classify_weborganizer*.py     independent topic classifier (WebOrganizer)
  diversity_check.py, topic_diversity.py        internal diversity of the final set
  contamination_check.py, contamination_check_fix.py   overlap with the agent's training questions and benchmarks
  run_openresearcher_batch.py         runs the questions through OpenResearcher-30B-A3B (copy into a clone of that repo)
  build_judge_packets.py              blind judge packets and answer keys
  build_split_packets.py              replaces multi-part items with atomic sub-items
  run_local_judge.py                  runs the judge locally with MLX
  score_judge_runs.py, aggregate_final.py, analyze_splitall.py, analyze_serper_and_split.py   scoring and reports
  judge_system_prompt_v2.txt          the judge prompt used for every reported result
  judge_system_prompt_v3.txt          a longer variant that lost to v2 under a pick rule fixed in advance
docker/                               images for the selection stage and the topic classifier
examples/split_spec_example.json      invented example of the split-spec format
```

Scripts use paths relative to the original project (`../outputs`). Run them from `data_collection/`.

## Input formats

`build_judge_packets.py` reads a workbook with three sheets.

- `Cases`: `case_id`, `prompt`, `as_of_date`, `evaluation_assumption`.
- `Core-Extra Rubric`: `case_id`, `item_id`, `item_type` (WIDTH or DEPTH), `requirement`, `tier` (core, extra or na),
  `tier_call`, `weight` (2 essential, 1 supporting), optionally `core_requirement` and `parent_item_id`.
- `WIDTH Equivalence v1`: `case_id`, `item_id`, `satisfying_paths`, `conditions_aligned_by_number`, `open_set`,
  `accept_other_if`.

Answers are read from a JSONL file with one object per case, `qid` and `messages` (chat format, last assistant
message is the answer). See `examples/split_spec_example.json` for the split format.

The judge returns one JSON object per case with `items` (verdict, rationale, borderline flag) and an
`accuracy` note. The format is defined at the end of `judge_system_prompt_v2.txt`.

## Scoring rule

The core score of a case is the weight of core items the judge passed, divided by the total weight of core items.
Extra items are graded and reported but never scored. Judge verdicts are the majority of three runs
(seeds 0, 1, 2). Tiers, weights and reference grades are never shown to the judge, which is also not told the name
of the system that wrote the answer.

## Requirements

- Selection stage: `data_collection/requirements.txt`, plus Ollama with `qwen2.5:7b` for the classifiers and a
  Hugging Face token approved for LMSYS-Chat-1M.
- Judging: Apple silicon, `mlx-lm`, `openpyxl`. The study used Muse Glimmer 30B (8-bit) at temperature 1.0,
  top-p 0.95, top-k 64.
- Agent runs: a clone of [OpenResearcher](https://github.com/TIGER-AI-Lab/OpenResearcher) and a search backend. The
  runner takes `--browser-backend serper` (needs `SERPER_API_KEY` in your environment) or `free`.

Keep secrets in a local `.env`, which is git-ignored. Do not send questions, checklists or answers to third-party
services if your data comes from a gated source.

## Limits

This is research code written for one study. It is not packaged, has no tests, and some scripts carry the
assumptions of the original project layout. The study's limitations are listed in the paper.

## Licence

MIT. See `LICENSE`.

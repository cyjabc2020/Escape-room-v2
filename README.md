# Trust Between AI Agents — Escape Room Survival Game

Code, settings, example run data, and analysis for the paper **"Trust Between
AI Agents: Measuring Formation, Breakage, and Recovery, with Implications for
Governing Multi-Agent Systems"** (Chen, 2026). Preprint: arXiv (link
forthcoming).

We propose a behavioral measure of trust between LLM agents based on costly
verification. In a four-agent cooperative survival game, checking a teammate's
work costs a coin and trusting a wrong answer can be fatal. One teammate (D)
is a scripted responder whose correctness follows a controlled failure
schedule; the trusting agents are drawn from six frontier model snapshots
(Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.1, GPT-5.4-mini, Gemini 3.1 Pro,
Gemini 2.5 Flash). Measured against a memoryless variant of the same model,
reduced verification of a proven partner provides an observable measure of
trust, and the framework tracks its formation, breakage, and recovery.

## Repository layout

```
code/
  src/             Game engine (agents, game loop, moderator, provider helpers)
  settings/        Experiment configs (model x schedule cells)
    main/                Smooth and 1-strike (recovery) cells
    ablation_d_schedule/ 2-strike, 3-strike, mid2strike, recur cells
    baseline_memoryless/ Memoryless (no-history) anchor cells
  scripts/         Analysis pipeline and the paper's derived data (CSVs)
  results/         EXAMPLE runs only (one per condition type; see below)
  question_banks/  Puzzle pools
  run_batch.sh     N replicates of one cell, one timestamped folder each
```

## Data availability

`code/results/` ships with five example runs (smooth, 1-strike, mid2strike,
a GPT-5.1 recovery run, and a memoryless anchor run) so the log format and
analysis pipeline can be inspected end to end. The full dataset behind the
paper (170 with-memory runs plus the memoryless anchor cells) is too large
for the repository; the paper's anchor-relative deltas are preserved in
`code/scripts/anchor_deltas.csv`, and the raw logs are available from the
author on request.

## Analysis pipeline

```bash
cd code
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd scripts
python anchor_stats.py             # anchor-relative deltas + cluster bootstrap CIs
                                   # (excludes games where D's scripted answer was wrong)
python driver_decomposition.py     # score drivers (Table 3)
python scenario_score_matrix.py    # scenario x model score matrix (Table 6)
python iid_rounds.py               # Fig. 1
python figure1_stacked.py          # Fig. 2
python volume_targeting_scatter.py # Fig. 3
python figure3_stacked.py          # Fig. 4
```

Figure scripts write to `code/scripts/figures/`. Note: run against the example
results these scripts produce example-sized cells; the shipped CSVs are the
canonical outputs computed over the full dataset.

## Run new sessions

```bash
cd code
# Put API keys in code/.env (only the providers you call are required):
#   OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...  REPLICATE_API_TOKEN=...
./run_batch.sh 10 settings/main/main_claude_opus_recovery.json
```

## Citation

```bibtex
@article{chen2026trustbetween,
  title  = {Trust Between {AI} Agents: Measuring Formation, Breakage, and
            Recovery, with Implications for Governing Multi-Agent Systems},
  author = {Chen, Yujiao},
  year   = {2026}
}
```

## License

Released under the MIT License (see `LICENSE`).

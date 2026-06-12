# Escape Room Survival Game — engine and analysis

This directory holds the game engine, model helpers, experiment runner, and the
analysis pipeline. See the top-level `README.md` for the project overview and the
accompanying paper.

The Escape Room Survival Game is a repeated cooperative environment in which four
agents must collaboratively assemble a shared password under mortal risk:
volunteering an incorrect password kills the volunteer, and collective inaction
kills a random agent. Each agent knows only its own puzzle and sees other puzzles
only as *capsules* (a puzzle index plus the list of endorsers, never the answer);
verifying a puzzle costs a coin and lets the agent solve it itself, endorsing a
matching capsule or opening a new one. One teammate (D) is a scripted responder
whose correctness follows a controlled failure schedule.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# API keys (only the providers you intend to call are required):
#   echo "OPENAI_API_KEY=..."     >> .env
#   echo "ANTHROPIC_API_KEY=..."  >> .env
#   echo "GEMINI_API_KEY=..."     >> .env
#   echo "REPLICATE_API_TOKEN=..." >> .env

# One replicate session (one model x schedule cell):
python -u src/run_experiment.py settings/main/main_claude_opus_recovery.json

# N replicates, each in its own timestamped folder:
./run_batch.sh 10 settings/main/main_claude_opus_recovery.json
```

Each `run_experiment.py` invocation runs a series of games with one fresh set of
puzzle assignments and writes a folder under `results/<experiment_name>_<timestamp>/`
containing one JSON per game plus aggregated CSVs.

## Analysis pipeline

`compute_metrics.py` builds the canonical `per_rep_metrics.csv` (one row per run).
`anchor_stats.py` computes the paper's anchor-relative deltas (trust formation and
culprit targeting) with cluster-bootstrap CIs, excluding games in which D's
scripted answer was wrong; `driver_decomposition.py` and `scenario_score_matrix.py`
produce the score tables; `iid_rounds.py`, `figure1_stacked.py`,
`volume_targeting_scatter.py`, and `figure3_stacked.py` produce the paper's
figures (written to `scripts/figures/`). The shipped CSVs in `scripts/` are the
canonical outputs computed over the full dataset; `results/` here contains
example runs only (see the top-level README).

## Layout

```
src/                 Core game engine
  run_experiment.py  Loads a settings file, runs games, writes CSVs
  agents/            Agent class with short-term and long-term memory
  games/             Game loop, prompt construction, capsule/verification logic
  utils/             Provider helpers (OpenAI, Anthropic, Gemini, Replicate),
                     settings loader, question bank, logger, answer checker
settings/            Experiment configs (main, ablation_d_schedule, baseline_memoryless)
scripts/             Analysis pipeline and derived data (CSVs)
results/             Example runs, one timestamped folder per run
```

## Model dispatch

`src/utils/replicate_helper.py` routes by model-name prefix: `openai:` to the
OpenAI helper, `anthropic:` to the Anthropic helper, `gemini:` to the Google
helper (auto-detecting the 2.5 vs 3.x family), `dummy` to the scripted teammate,
and any other name to the Replicate platform.

## License

Released under the MIT License unless otherwise noted.

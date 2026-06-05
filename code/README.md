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

```bash
cd scripts
./regenerate.sh   # compute_metrics -> statistics -> clustering -> figures -> PDF
```

`compute_metrics.py` builds the canonical `per_rep_metrics.csv` (one row per run)
that every downstream analysis consumes: `statistical_models.py` (pairwise tests
and the schedule logit), `clustering_analysis.py` and `clustering_robustness.py`
(verification-profile clusters), `bayesian_baseline.py` (the normative reference),
and `paper_figures.py` (figures). It standardizes sampling to the earliest 10 runs
per rough-start cell and 5 per smooth baseline.

## Layout

```
src/                 Core game engine
  run_experiment.py  Loads a settings file, runs games, writes CSVs
  agents/            Agent class with short-term and long-term memory
  games/             Game loop, prompt construction, capsule/verification logic
  utils/             Provider helpers (OpenAI, Anthropic, Gemini, Replicate),
                     settings loader, question bank, logger, answer checker
settings/            Experiment configs, grouped by family (main, ablation_*)
scripts/             Analysis pipeline and regenerate.sh
results/             One timestamped folder per run
```

## Model dispatch

`src/utils/replicate_helper.py` routes by model-name prefix: `openai:` to the
OpenAI helper, `anthropic:` to the Anthropic helper, `gemini:` to the Google
helper (auto-detecting the 2.5 vs 3.x family), `dummy` to the scripted teammate,
and any other name to the Replicate platform.

## License

Released under the MIT License unless otherwise noted.

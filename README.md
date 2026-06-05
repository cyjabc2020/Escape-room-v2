# Escape Room Survival Game — Cross-Snapshot Trust Recovery

Code, settings, run data, and analysis for the paper **"How LLM Agents Recover
Trust: Empirical Verification Profiles and Concentration-Driven Scarring across
Six Frontier Model Snapshots"** (Chen, 2026). Preprint: [arXiv link forthcoming].

We study how an LLM agent treats a teammate that fails early and then becomes
reliable. In a four-agent cooperative survival game, one teammate (D) is a
scripted responder that gives wrong answers on a controlled schedule before
becoming dependable. We vary which of six frontier model snapshots plays the
trusting agent — Claude Opus 4.6, Claude Sonnet 4.6, GPT-5.1, GPT-5.4-mini,
Gemini 3.1 Pro, Gemini 2.5 Flash — and how the teammate's failures are spaced,
and we measure how much the agent keeps verifying the failed teammate
(targeted caution) versus the rest of the team (spillover distrust).

## Repository layout

```
code/      Game engine, model helpers, experiment runner, and analysis pipeline
  src/             Cooperative survival game (agents, game loop, moderator, helpers)
  settings/        Experiment configs (model x schedule cells)
  scripts/         Analysis pipeline + regenerate.sh
  results/         Raw per-run logs (one timestamped folder per run)
paper/     LaTeX source, figures, and compiled PDF
```

## Reproduce the paper's numbers and figures

```bash
cd code
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cd scripts
./regenerate.sh      # rebuilds metrics, statistics, clustering, figures, and the PDF
```

`regenerate.sh` reads `code/results/`, standardizes sampling (the earliest 10
runs per rough-start cell and 5 per smooth baseline), and writes
`per_rep_metrics.csv` plus the statistics, clustering, and figure outputs the
paper draws on.

## Run new sessions

```bash
cd code
# Put API keys in code/.env:
#   OPENAI_API_KEY=...  ANTHROPIC_API_KEY=...  GEMINI_API_KEY=...  REPLICATE_API_TOKEN=...
./run_batch.sh 10 settings/main/main_claude_opus_recovery.json
```

## Citation

```bibtex
@article{chen2026recovertrust,
  title  = {How {LLM} Agents Recover Trust: Empirical Verification Profiles and
            Concentration-Driven Scarring across Six Frontier Model Snapshots},
  author = {Chen, Yujiao},
  year   = {2026}
}
```

## License

Released under the MIT License unless otherwise noted.

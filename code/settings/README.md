# Experiment Settings

This folder contains the eight JSON configurations corresponding to the
2 × 2 × 2 design reported in the paper:

- **Scenario:** `rough_start` (partner gives the wrong answer in game 1, then is reliable) vs. `trust_converge` (partner is reliable from game 1).
- **Reasoning effort:** `high` vs. `none` (set via the GPT-5.1 `reasoning_effort` parameter).
- **Reflective memory:** `reflection` (agents generate post-game reflections that persist across games) vs. `none`.

All eight files share the same structure: 4 players (3 GPT-5.1 agents + 1 dummy
partner), 10–11 games per run, `basic_arithmetic.csv` as the puzzle source, and
`max_allowed` memory window (the `_max` suffix in the filename).

| File | Scenario | Reasoning | Reflection | Games |
|---|---|---|---|---|
| `rough_start_high_reflection_max.json` | rough_start | high | evolving | 11 |
| `rough_start_high_none_max.json` | rough_start | high | none | 11 |
| `rough_start_none_reflection_max.json` | rough_start | none | evolving | 11 |
| `rough_start_none_none_max.json` | rough_start | none | none | 11 |
| `trust_converge_high_reflection_max.json` | trust_converge | high | evolving | 10 |
| `trust_converge_high_none_max.json` | trust_converge | high | none | 10 |
| `trust_converge_none_reflection_max.json` | trust_converge | none | evolving | 10 |
| `trust_converge_none_none_max.json` | trust_converge | none | none | 10 |

## How to run

```bash
python src/run_experiment.py settings/rough_start_high_reflection_max.json
```

Each invocation produces one *replicate session* (10–11 games with a freshly
sampled puzzle set) and writes it to a new
`results/<experiment_name>_<timestamp>/` folder. The `log_directory` in
`output_settings` is automatically rewritten from `game_logs` to `results/`
when the settings file lives in `settings/`.

The paper's Table 1 reports mean ± SE across **7–14 replicate sessions per
condition**; this repo ships **one canonical session per condition**. To
exactly reproduce Table 1, run each settings file 7–14 times.

## Naming note

The paper calls these conditions **"trust-recovery"** and **"smooth start"**;
in the codebase they are `rough_start` and `trust_converge` respectively. This
is purely a legacy of how the experiments were named during development.

## Customizing

Copy any of the eight files and edit:

- `game_settings.num_players`, `num_games`, `question_sets`
- `player_models` — a list of Replicate model identifiers, plus `"dummy"` for the controlled partner
- `reasoning_effort_config` — per-agent `"none" | "low" | "medium" | "high"`
- `memory_config` — per-agent `mode` (`"none" | "last_n" | "current_game_only" | "max_allowed"`), `window`, `reflection`, `persist_across_games`
- `dummy_config[<agent_id>].correctness_list` — per-game booleans controlling whether the dummy gives the correct answer in that game

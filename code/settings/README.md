# Experiment Settings

JSON configurations for the conditions reported in the paper. Every file uses
the same structure: 4 players (3 LLM agents A/B/C of the snapshot under test +
1 scripted dummy partner D), arithmetic puzzles, and a `dummy_config` whose
`correctness_list` controls D's per-game reliability (the D-failure schedule).

## Folders

```
main/                 Smooth and 1-strike cells, one pair per snapshot
ablation_d_schedule/  2-strike, 3-strike, mid2strike, recur cells
baseline_memoryless/  Memoryless (no-history) anchor cells, D-correct and D-wrong
```

## Snapshots

| Settings prefix | Snapshot |
|---|---|
| `*claude_opus*` / `*opus*` | claude-opus-4-6 (Adaptive thinking) |
| `*claude_sonnet*` / `*sonnet*` | claude-sonnet-4-6 (Adaptive thinking) |
| `*gpt5*` / `*gpt5_1*` | gpt-5.1 (reasoning_effort="high") |
| `*gpt5_mini*` | gpt-5.4-mini-2026-03-17 (reasoning_effort="high") |
| `*gemini*` (no suffix) / `*gemini_pro*` | gemini-3.1-pro-preview |
| `*gemini_flash*` | gemini-2.5-flash |

## D-failure schedules

| Schedule | `correctness_list` (T = correct, F = wrong) | Where |
|---|---|---|
| smooth | [T, ..., T] (all correct, 11 games) | `main/*_smooth.json` |
| 1-strike | [F,T,T,T,T,T,T,T,T,T,T] | `main/*_recovery.json` |
| 2-strike | [F,F,T,T,T,T,T,T,T,T,T] | `ablation_d_schedule/*_2strike.json` |
| 3-strike | [F,F,F,T,T,T,T,T,T,T,T] | `ablation_d_schedule/*_3strike.json` |
| mid2strike | [T,T,T,F,F,T,T,T,T,T,T] | `ablation_d_schedule/*_mid2strike.json` |
| recur | [F,T,T,T,F,T,T,T,T,T,T] | `ablation_d_schedule/*_recur.json` |

The memoryless anchor cells (`baseline_memoryless/`) disable cross-game memory
(`persist_across_games: false`), so each game is an independent draw with no
shared history; `*_dcorrect.json` keeps D correct throughout (the paper's
anchor) and `*_dwrong.json` keeps D wrong throughout (the secondary
reference). The `*_always.json` files in `ablation_d_schedule/` (D never
reliable) are not used in the paper.

The paper's sampling targets were n = 10 runs per perturbed cell, n = 5 per
smooth cell, and at least 50 independent games per memoryless anchor cell.

## Naming notes

`main/*_recovery.json` is the paper's **1-strike** condition (legacy name from
development). Gemini 3.1 Pro files are `main_gemini_*.json` with no "pro" in
the name; Gemini 2.5 Flash files are `main_gemini_flash_*.json`.

## How to run

```bash
# one run (one model x schedule cell):
python src/run_experiment.py settings/main/main_claude_opus_recovery.json

# N runs, each in its own timestamped results folder:
./run_batch.sh 10 settings/main/main_claude_opus_recovery.json
```

## Customizing

Copy any file and edit:

- `game_settings.num_players`, `num_games`, `question_sets`
- `player_models` — model identifiers (`openai:`, `anthropic:`, `gemini:` prefixes route to the direct provider APIs; `"dummy"` is the scripted partner)
- `reasoning_effort_config` — per-agent `"none" | "low" | "medium" | "high" | "adaptive"`
- `memory_config` — per-agent `mode`, `window`, `reflection`, `persist_across_games`
- `dummy_config[<agent_id>].correctness_list` — per-game booleans for D's schedule

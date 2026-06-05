# -*- coding: utf-8 -*-
"""
Track and plot individual player decisions across games.

This script takes a simulation prefix, finds all matching result folders,
and creates Sankey-style flow diagrams showing decision transitions from game 2 to game 10.

Usage:
    python decision_track_record.py

Example:
    # Edit the SIMULATION_PREFIX variable in main() to specify which simulation to analyze
    SIMULATION_PREFIX = 'rough_start_high_none_max'
"""

import pandas as pd
import matplotlib
matplotlib.rcParams.update({
    'font.size':            25,
    'axes.titlesize':       22,
    'axes.labelsize':       18,
    'xtick.labelsize':      14,
    'ytick.labelsize':      14,
    'legend.fontsize':      35,
    'axes.linewidth':       1.2,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path as MplPath
import numpy as np
from pathlib import Path
import json

# Try plotly for HTML export; fall back gracefully
try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# Decision type to nominal value mapping
DECISION_VALUE_MAP = {
    "volunteer": 6,
    "verify:Q4": 5,
    "verify:Q3": 4,
    "verify:Q2": 3,
    "verify:Q1": 2,
    "pass": 1
}


def find_matching_folders(results_dir: Path, prefix: str):
    """Find all subfolders in results_dir that start with the given prefix."""
    matching_folders = []
    if not results_dir.exists():
        print(f"   Warning: Results directory not found: {results_dir}")
        return matching_folders
    for folder in results_dir.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            matching_folders.append(folder)
    return sorted(matching_folders)


def categorize_decision_simple(decision_str):
    """Categorize an agent decision into a simple string format."""
    if pd.isna(decision_str) or decision_str == 'N/A':
        return None
    if decision_str == 'Pass':
        return 'pass'
    if decision_str.startswith('Verify:'):
        verify_target = decision_str.replace('Verify:', '').strip()
        if verify_target and verify_target.startswith('Q'):
            return f'verify:{verify_target}'
        return 'pass'
    if decision_str.startswith('Volunteer:'):
        return 'volunteer'
    return 'pass'


def map_decision_to_value(decision_category):
    """Map a decision category to its nominal value."""
    if decision_category is None:
        return None
    return DECISION_VALUE_MAP.get(decision_category, None)


def _collect_decisions(results_dir, prefix, matching_folders):
    """Collect and aggregate decision data from simulation folders."""
    all_decisions = []
    for sim_idx, folder in enumerate(matching_folders):
        csv_file = folder / "game_data_agent_decisions.csv"
        if not csv_file.exists():
            print(f"   Warning: CSV not found in {folder.name}")
            continue
        print(f"\n   Processing simulation {sim_idx + 1}: {folder.name}")
        df = pd.read_csv(csv_file)
        unique_game_ids = sorted(df['game_id'].unique())
        game_id_to_number = {gid: idx + 1 for idx, gid in enumerate(unique_game_ids)}
        df['game_number'] = df['game_id'].map(game_id_to_number)
        df_filtered = df[(df['game_number'] >= 2) & (df['game_number'] <= 10)].copy()
        if len(df_filtered) == 0:
            continue
        df_filtered['decision_category'] = df_filtered['agent_decision'].apply(categorize_decision_simple)
        df_filtered['decision_value'] = df_filtered['decision_category'].apply(map_decision_to_value)
        agents = sorted(df_filtered['agent_id'].unique())
        print(f"   Found {len(agents)} agents: {agents}")
        df_filtered = df_filtered[df_filtered['agent_id'].isin(['A', 'B', 'C'])].copy()
        print(f"   Excluding agent D, keeping only A, B, C")
        df_first_decision = df_filtered[df_filtered['round'] == 1].copy()
        df_first_decision['simulation_idx'] = sim_idx
        df_first_decision['sim_agent_id'] = df_first_decision.apply(
            lambda row: f"sim{sim_idx}_agent{row['agent_id']}", axis=1
        )
        all_decisions.append(df_first_decision[['sim_agent_id', 'game_number', 'decision_category']])
    if not all_decisions:
        return None
    combined = pd.concat(all_decisions, ignore_index=True)
    print(f"\n   Total decision records: {len(combined)}")
    print(f"   Unique player-tracks: {combined['sim_agent_id'].nunique()}")
    return combined


def _build_transitions(combined_decisions, decision_types, games):
    """Build transition counts between consecutive games."""
    transitions = {}  # (game, from_dec, to_dec) -> count
    for i in range(len(games) - 1):
        game, next_game = games[i], games[i + 1]
        curr = combined_decisions[combined_decisions['game_number'] == game]
        nxt = combined_decisions[combined_decisions['game_number'] == next_game]
        merged = curr.merge(nxt, on='sim_agent_id', suffixes=('_c', '_n'))
        counts = merged.groupby(['decision_category_c', 'decision_category_n']).size().reset_index(name='count')
        for _, row in counts.iterrows():
            fd, td, cnt = row['decision_category_c'], row['decision_category_n'], row['count']
            if pd.isna(fd) or pd.isna(td):
                continue
            transitions[(game, fd, td)] = transitions.get((game, fd, td), 0) + cnt
    return transitions


def _draw_bezier(ax, x0, y0, x1, y1, height, color, alpha=0.35):
    """Draw a filled bezier curve (flow band) between two nodes."""
    # Control point offset for smooth curve
    dx = (x1 - x0) * 0.4
    verts_top = [
        (x0, y0 + height / 2),
        (x0 + dx, y0 + height / 2),
        (x1 - dx, y1 + height / 2),
        (x1, y1 + height / 2),
    ]
    verts_bottom = [
        (x1, y1 - height / 2),
        (x1 - dx, y1 - height / 2),
        (x0 + dx, y0 - height / 2),
        (x0, y0 - height / 2),
    ]
    verts = verts_top + verts_bottom + [verts_top[0]]
    codes = [
        MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY
    ]
    path = MplPath(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=color, edgecolor='none', alpha=alpha)
    ax.add_patch(patch)


def plot_decision_tracks_matplotlib(combined_decisions, prefix, output_dir):
    """
    Create a matplotlib-based Sankey diagram showing decision flow across games.
    Produces publication-quality PNG with large, legible fonts.
    """
    decision_types = ['volunteer', 'verify:Q4', 'verify:Q3', 'verify:Q2', 'verify:Q1', 'pass']
    decision_colors = {
        'pass': '#93c5fd',
        'verify:Q1': '#fca5a5',
        'verify:Q2': '#f87171',
        'verify:Q3': '#ef4444',
        'verify:Q4': '#dc2626',
        'volunteer': '#86efac'
    }
    decision_labels = {
        'pass': 'Pass',
        'verify:Q1': 'V:Q1',
        'verify:Q2': 'V:Q2',
        'verify:Q3': 'V:Q3',
        'verify:Q4': 'V:Q4',
        'volunteer': 'Vol',
    }

    games = list(range(2, 11))
    transitions = _build_transitions(combined_decisions, decision_types, games)

    # Count totals per (game, decision) for node sizing
    node_counts = {}
    for game in games:
        game_data = combined_decisions[combined_decisions['game_number'] == game]
        for dt in decision_types:
            cnt = len(game_data[game_data['decision_category'] == dt])
            node_counts[(game, dt)] = cnt

    # Normalize: find max count for scaling
    max_count = max(node_counts.values()) if node_counts else 1
    if max_count == 0:
        max_count = 1

    # Layout parameters
    fig, ax = plt.subplots(figsize=(30, 15.4))
    x_margin = 1.0
    x_spacing = 2.2
    node_width = 0.35
    y_total = 10.0
    y_pad = 0.3  # padding between nodes

    # Compute node positions: for each game column, stack nodes vertically
    node_positions = {}  # (game, decision) -> (x_center, y_center, height)

    for gi, game in enumerate(games):
        x = x_margin + gi * x_spacing

        # Compute heights proportional to counts
        total_count = sum(node_counts.get((game, dt), 0) for dt in decision_types)
        available_height = y_total - y_pad * (len(decision_types) - 1)

        y_cursor = y_total
        for dt in decision_types:
            cnt = node_counts.get((game, dt), 0)
            if total_count > 0:
                h = max((cnt / total_count) * available_height, 0.08)
            else:
                h = 0.08
            y_center = y_cursor - h / 2
            node_positions[(game, dt)] = (x, y_center, h)
            y_cursor -= (h + y_pad)

    # Draw flow bands (transitions)
    # Track how much vertical space has been used at each node for stacking flows
    node_offsets_out = {k: 0.0 for k in node_positions}
    node_offsets_in = {k: 0.0 for k in node_positions}

    for gi in range(len(games) - 1):
        game = games[gi]
        next_game = games[gi + 1]
        for dt_from in decision_types:
            for dt_to in decision_types:
                cnt = transitions.get((game, dt_from, dt_to), 0)
                if cnt == 0:
                    continue

                # Source node
                sx, sy, sh = node_positions[(game, dt_from)]
                src_total = node_counts.get((game, dt_from), 1)
                band_h_src = (cnt / max(src_total, 1)) * sh

                # Target node
                tx, ty, th = node_positions[(next_game, dt_to)]
                tgt_total = node_counts.get((next_game, dt_to), 1)
                band_h_tgt = (cnt / max(tgt_total, 1)) * th

                # Use average band height
                band_h = (band_h_src + band_h_tgt) / 2

                # Compute vertical positions using offsets
                y0 = sy + sh / 2 - node_offsets_out[(game, dt_from)] - band_h_src / 2
                y1 = ty + th / 2 - node_offsets_in[(next_game, dt_to)] - band_h_tgt / 2

                node_offsets_out[(game, dt_from)] += band_h_src
                node_offsets_in[(next_game, dt_to)] += band_h_tgt

                color = decision_colors[dt_from]
                _draw_bezier(ax, sx + node_width / 2, y0,
                             tx - node_width / 2, y1,
                             band_h, color, alpha=0.35)

    # Draw nodes (rectangles)
    for (game, dt), (x, y, h) in node_positions.items():
        cnt = node_counts.get((game, dt), 0)
        if cnt == 0:
            continue
        color = decision_colors[dt]
        rect = mpatches.FancyBboxPatch(
            (x - node_width / 2, y - h / 2), node_width, h,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='#333333', linewidth=0.8
        )
        ax.add_patch(rect)

        # Label on node (only show if big enough)
        if h > 0.25:
            label = decision_labels.get(dt, dt)
            ax.text(x, y, label, ha='center', va='center',
                    fontsize=36, fontweight='bold', color='#1a1a1a')

    # Game labels along top
    for gi, game in enumerate(games):
        x = x_margin + gi * x_spacing
        ax.text(x, y_total + 0.6, f'G{game}', ha='center', va='bottom',
                fontsize=42, fontweight='bold', color='#333333')

    # Title
    parts = prefix.split('_')
    if len(parts) >= 3:
        short = f"{parts[-3]}_{parts[-2]}"
    else:
        short = prefix
    ax.set_title(f'Decision Flow Across Games (Round 1) — {short}',
                 fontsize=60, fontweight='bold', pad=80)

    # Legend at top, between title and game labels
    legend_elements = [
        mpatches.Patch(facecolor='#93c5fd', edgecolor='#333', label='Pass'),
        mpatches.Patch(facecolor='#fca5a5', edgecolor='#333', label='Verify'),
        mpatches.Patch(facecolor='#86efac', edgecolor='#333', label='Volunteer'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', fontsize=58,
              framealpha=0.9, edgecolor='#cccccc', fancybox=True,
              ncol=3, bbox_to_anchor=(0.5, 1.08))

    # Clean up axes
    ax.set_xlim(0, x_margin + len(games) * x_spacing)
    ax.set_ylim(-1, y_total + 2.0)
    ax.axis('off')

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'decision_sankey_{prefix}.png'
    plt.savefig(output_file, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"\n   ✅ Matplotlib Sankey saved to: {output_file}")
    plt.close()


def plot_decision_tracks_plotly(combined_decisions, prefix, output_dir):
    """Create a Plotly-based Sankey diagram (HTML + optional PNG)."""
    if not HAS_PLOTLY:
        print("   ⚠️  Plotly not available, skipping HTML export")
        return

    decision_types = ['volunteer', 'verify:Q4', 'verify:Q3', 'verify:Q2', 'verify:Q1', 'pass']
    decision_colors = {
        'pass': '#93c5fd',
        'verify:Q1': '#fca5a5',
        'verify:Q2': '#fca5a5',
        'verify:Q3': '#fca5a5',
        'verify:Q4': '#fca5a5',
        'volunteer': '#86efac'
    }
    decision_short = {
        'pass': 'Pass',
        'verify:Q1': 'V:Q1',
        'verify:Q2': 'V:Q2',
        'verify:Q3': 'V:Q3',
        'verify:Q4': 'V:Q4',
        'volunteer': 'Vol',
    }

    games = list(range(2, 11))

    # Create nodes
    node_labels = []
    node_colors = []
    node_idx_map = {}
    idx = 0
    for game in games:
        for decision in decision_types:
            short = decision_short.get(decision, decision)
            node_labels.append(f"G{game} {short}")
            node_colors.append(decision_colors[decision])
            node_idx_map[(game, decision)] = idx
            idx += 1

    # Create links
    transitions = _build_transitions(combined_decisions, decision_types, games)
    sources, targets, values, link_colors = [], [], [], []
    for (game, fd, td), cnt in transitions.items():
        si = node_idx_map.get((game, fd))
        ti = node_idx_map.get((game + 1, td))
        if si is not None and ti is not None:
            sources.append(si)
            targets.append(ti)
            values.append(cnt)
            hc = decision_colors[fd]
            r, g, b = int(hc[1:3], 16), int(hc[3:5], 16), int(hc[5:7], 16)
            link_colors.append(f'rgba({r},{g},{b},0.4)')

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=25, thickness=30,
            line=dict(color="black", width=0.8),
            label=node_labels, color=node_colors
        ),
        link=dict(source=sources, target=targets, value=values, color=link_colors),
        textfont=dict(size=18, color="black"),
    )])
    fig.update_layout(
        title=dict(
            text=f"Decision Flow Across Games 2-10<br>{prefix}",
            font=dict(size=28, color="black"),
            x=0.5, xanchor='center',
        ),
        font=dict(size=20),
        height=1000, width=1800,
        margin=dict(l=40, r=40, t=100, b=40),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'decision_sankey_{prefix}.html'
    fig.write_html(str(output_file))
    print(f"\n   ✅ Plotly Sankey HTML saved to: {output_file}")

    try:
        output_png = output_dir / f'decision_sankey_{prefix}_plotly.png'
        fig.write_image(str(output_png), width=1800, height=1000, scale=2)
        print(f"   ✅ Plotly PNG saved to: {output_png}")
    except Exception as e:
        print(f"   ⚠️  Could not save Plotly PNG: {e}")


def plot_decision_tracks(results_dir: Path, prefix: str, output_dir: Path):
    """
    Main entry: create Sankey diagrams showing decision flow across games 2-10.
    Produces both matplotlib PNG (always) and Plotly HTML (if available).
    """
    print(f"\n{'='*80}")
    print(f"Processing: {prefix}")
    print(f"{'='*80}")

    matching_folders = find_matching_folders(results_dir, prefix)
    if not matching_folders:
        print(f"   ❌ No matching folders found for prefix: {prefix}")
        return

    print(f"   Found {len(matching_folders)} matching simulation(s):")
    for folder in matching_folders:
        print(f"     - {folder.name}")

    combined_decisions = _collect_decisions(results_dir, prefix, matching_folders)
    if combined_decisions is None:
        print(f"   ❌ No valid data found")
        return

    # Always produce matplotlib PNG (works everywhere, publication quality)
    plot_decision_tracks_matplotlib(combined_decisions, prefix, output_dir)

    # Also produce Plotly HTML if available (interactive)
    plot_decision_tracks_plotly(combined_decisions, prefix, output_dir)


def main():
    """Main entry point."""
    # =========================================================================
    # CONFIGURATION: Edit this to specify which simulation to analyze
    # =========================================================================
    SIMULATION_PREFIX = 'rough_start_high_reflection_max'
    # =========================================================================

    results_dir = Path('results')
    output_dir = Path('analysis_output')

    print("=" * 80)
    print("DECISION TRACK RECORD: Individual Player Decisions Across Games")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Simulation prefix: {SIMULATION_PREFIX}")

    plot_decision_tracks(results_dir, SIMULATION_PREFIX, output_dir)

    print("\n" + "=" * 80)
    print(f"✅ Analysis complete!")
    print(f"📁 All files saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

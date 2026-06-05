# -*- coding: utf-8 -*-
"""
Aggregate and plot agent decisions across multiple simulations with matching prefixes.

This script takes a list of simulation prefixes, finds all matching result folders,
combines their data, and creates aggregated stacked bar charts.

Usage:
    python analyze_all.py

Example:
    # The CONDITION_GROUPS dict in main() specifies which simulations to analyze.
    # By default it processes the eight `_max` conditions reported in the paper:
    #   rough_start_{high,none}_{none,reflection}_max
    #   trust_converge_{high,none}_{none,reflection}_max
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Global figure / font settings for publication-quality (NeurIPS) figures
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    'font.size':            18,
    'axes.titlesize':       22,
    'axes.labelsize':       20,
    'xtick.labelsize':      16,
    'ytick.labelsize':      16,
    'legend.fontsize':      16,
    'figure.titlesize':     24,
    'lines.linewidth':      2.5,
    'lines.markersize':     8,
    'axes.linewidth':       1.2,
    'xtick.major.width':    1.0,
    'ytick.major.width':    1.0,
    'xtick.major.size':     5,
    'ytick.major.size':     5,
})


def read_dummy_agents_from_settings(folder_path):
    """
    Read dummy agents and their correctness info from experiment_settings.json.

    Args:
        folder_path: Path to experiment folder

    Returns:
        Tuple of (set of dummy agent IDs, dict of dummy_id -> correctness_list)
    """
    settings_file = folder_path / "experiment_settings.json"
    dummy_agents = set()
    dummy_correctness = {}

    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)

            # Check dummy_config
            dummy_config = settings.get('dummy_config', {})
            if dummy_config:
                dummy_agents = set(dummy_config.keys())
                # Extract correctness_list for each dummy
                for dummy_id, config in dummy_config.items():
                    if 'correctness_list' in config:
                        dummy_correctness[dummy_id] = config['correctness_list']

            # Also check player_models for "dummy"
            player_models = settings.get('player_models', [])
            agent_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            for i, model in enumerate(player_models):
                if model == 'dummy' and i < len(agent_letters):
                    dummy_agents.add(agent_letters[i])

        except Exception as e:
            print(f"   Warning: Could not read dummy agents from settings: {e}")

    return dummy_agents, dummy_correctness


def get_num_players_from_settings(folder_path):
    """
    Read number of players from experiment_settings.json.

    Args:
        folder_path: Path to experiment folder

    Returns:
        Number of players, or None if not found
    """
    settings_file = folder_path / "experiment_settings.json"

    if settings_file.exists():
        try:
            with open(settings_file, 'r') as f:
                settings = json.load(f)
            return settings.get('game_settings', {}).get('num_players', None)
        except:
            pass

    return None


def categorize_decision(decision_str, dummy_agents=None):
    """
    Categorize an agent decision into one of the decision types.

    Args:
        decision_str: The decision string (e.g., "Pass", "Verify: Q1", "Volunteer: Q1-A, Q2-B, Q3-C")
        dummy_agents: Set of dummy agent IDs (if any)

    Returns:
        Category string: 'pass', 'verify_dummy', 'verify_other', 'volunteer_with_dummy', 'volunteer_correct'
    """
    # Skip N/A decisions (dead agents)
    if pd.isna(decision_str) or decision_str == 'N/A':
        return None

    if decision_str == 'Pass':
        return 'pass'

    if decision_str.startswith('Verify:'):
        # Extract which question is being verified (e.g., "Verify: Q1" -> "Q1")
        verify_target = decision_str.replace('Verify:', '').strip()

        # Check if the verified question belongs to a dummy
        # Question format is "Q1", "Q2", etc.
        # We need to map questions to agents (Q1->A, Q2->B, Q3->C, etc.)
        if verify_target and verify_target.startswith('Q'):
            try:
                question_num = int(verify_target[1:])
                # Map question number to agent (1->A, 2->B, 3->C, etc.)
                agent_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                if question_num <= len(agent_letters):
                    verified_agent = agent_letters[question_num - 1]
                    if dummy_agents and verified_agent in dummy_agents:
                        return 'verify_dummy'
            except (ValueError, IndexError):
                pass

        return 'verify_other'

    if decision_str.startswith('Volunteer:'):
        # Extract the capsules being used (e.g., "Volunteer: Q1-A, Q2-B, Q3-C")
        capsules_str = decision_str.replace('Volunteer:', '').strip()

        # Check if any capsule comes from a dummy agent
        # Capsules are in format "Q1-A", "Q2-B", "Q3-C"
        if dummy_agents:
            for dummy_id in dummy_agents:
                # Map agent to question number (A->1, B->2, C->3, etc.)
                agent_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                if dummy_id in agent_letters:
                    question_num = agent_letters.index(dummy_id) + 1
                    # Check for dummy's capsule in the format "Q3-C" (exact match)
                    dummy_capsule = f'Q{question_num}-{dummy_id}'
                    if dummy_capsule in capsules_str:
                        return 'volunteer_with_dummy'

        return 'volunteer_correct'

    return 'pass'


def find_matching_folders(results_dir: Path, prefix: str):
    """
    Find all subfolders in results_dir that start with the given prefix.

    Args:
        results_dir: Path to results directory
        prefix: Prefix to match

    Returns:
        List of matching folder paths
    """
    matching_folders = []

    if not results_dir.exists():
        print(f"   Warning: Results directory not found: {results_dir}")
        return matching_folders

    for folder in results_dir.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            matching_folders.append(folder)

    return sorted(matching_folders)


def aggregate_and_plot(results_dir: Path, prefix: str, output_dir: Path):
    """
    Aggregate data from all matching simulations and create a plot.

    Args:
        results_dir: Path to results directory
        prefix: Prefix to match simulation folders
        output_dir: Directory to save output plot

    Returns:
        Tuple of (avg_coin_df, per_sim_coin_df, decision_pivot, dummy_correctness, num_games) or (None, None, None, None, None) if error
    """
    print(f"\n{'='*80}")
    print(f"Processing: {prefix}")
    print(f"{'='*80}")

    # Find matching folders
    matching_folders = find_matching_folders(results_dir, prefix)

    if not matching_folders:
        print(f"   ❌ No matching folders found for prefix: {prefix}")
        return None, None, None

    print(f"   Found {len(matching_folders)} matching simulation(s):")
    for folder in matching_folders:
        print(f"     - {folder.name}")

    # Read dummy config from first matching folder
    first_folder = matching_folders[0]
    dummy_agents, dummy_correctness = read_dummy_agents_from_settings(first_folder)
    has_dummy = len(dummy_agents) > 0

    if has_dummy:
        print(f"   Detected dummy agents from settings: {sorted(dummy_agents)}")
    else:
        print(f"   No dummy agents in settings")

    # Get number of players
    num_players = get_num_players_from_settings(first_folder)

    # Load CSV files from each simulation separately and assign game numbers
    all_sim_data = []
    all_coin_data = []  # Store coin data for each simulation
    num_games_per_sim = None

    for sim_idx, folder in enumerate(matching_folders):
        csv_file = folder / "game_data_agent_decisions.csv"
        if not csv_file.exists():
            print(f"   Warning: CSV not found in {folder.name}")
            continue

        df = pd.read_csv(csv_file)
        df['simulation_id'] = sim_idx

        # Get unique game IDs in chronological order for this simulation
        unique_game_ids = sorted(df['game_id'].unique())

        # Map game_id to game_number (1-indexed position in chronological order)
        game_id_to_number = {game_id: idx + 1 for idx, game_id in enumerate(unique_game_ids)}
        df['game_number'] = df['game_id'].map(game_id_to_number)

        # Calculate total coins from all alive players at end of each game
        # Find the last round for each game
        last_rounds = df.groupby('game_id')['round'].max().reset_index()
        last_rounds.columns = ['game_id', 'last_round']

        df_with_last = df.merge(last_rounds, on='game_id')
        df_last_round = df_with_last[df_with_last['round'] == df_with_last['last_round']].copy()

        # Filter for alive players only
        df_alive = df_last_round[df_last_round['agent_status_end_of_round'] == 'ALIVE'].copy()

        # Calculate total coins per game (sum of all alive players)
        coins_per_game = df_alive.groupby('game_number')['coins_end_of_round'].sum().reset_index()
        coins_per_game.columns = ['game_number', 'total_coins']

        # Create complete grid for all game numbers (1 to len(unique_game_ids))
        # This ensures games with 0 survivors are represented as 0
        complete_game_grid = pd.DataFrame({'game_number': range(1, len(unique_game_ids) + 1)})
        coins_per_game = complete_game_grid.merge(coins_per_game, on='game_number', how='left')
        coins_per_game['total_coins'] = coins_per_game['total_coins'].fillna(0)

        coins_per_game['simulation_id'] = sim_idx
        all_coin_data.append(coins_per_game)

        # Categorize decisions for the bar chart
        df['decision_category'] = df['agent_decision'].apply(
            lambda x: categorize_decision(x, dummy_agents)
        )

        # Filter out None categories (N/A decisions from dead agents)
        df_valid = df[df['decision_category'].notna()].copy()

        # Filter out dummy agents' decisions - only count non-dummy player decisions
        if dummy_agents:
            df_valid = df_valid[~df_valid['agent_id'].isin(dummy_agents)].copy()
            print(f"   Excluding dummy agents: {sorted(dummy_agents)}")

        # Track number of games (should be same across simulations)
        if num_games_per_sim is None:
            num_games_per_sim = len(unique_game_ids)
        elif len(unique_game_ids) != num_games_per_sim:
            print(f"   Warning: {folder.name} has {len(unique_game_ids)} games, expected {num_games_per_sim}")

        all_sim_data.append(df_valid)

    if not all_sim_data:
        print(f"   ❌ No valid CSV files found for prefix: {prefix}")
        return None, None, None

    # Combine all simulation data
    combined_df = pd.concat(all_sim_data, ignore_index=True)
    print(f"   📊 Loaded {len(combined_df)} total records from {len(all_sim_data)} simulation(s)")

    # For rough_start, filter to only first 10 games
    if 'rough_start' in prefix:
        combined_df = combined_df[combined_df['game_number'] <= 10].copy()
        num_games_per_sim = min(10, num_games_per_sim) if num_games_per_sim else 10
        print(f"   🔍 Filtered to first 10 games for rough_start simulation")

    # Infer num_players from data if not found in settings
    if num_players is None:
        num_players = combined_df['agent_id'].nunique()

    print(f"   Number of players: {num_players} (max rounds per game)")
    max_rounds_per_game = num_players
    # Calculate actual number of games from filtered data
    num_games = combined_df['game_number'].nunique()
    print(f"   Number of games in plots: {num_games}")

    # Group by simulation, game_number, round, and decision_category to count
    decision_counts = combined_df.groupby(
        ['simulation_id', 'game_number', 'round', 'decision_category']
    ).size().reset_index(name='count')
    print(f"\n   STEP 1: Decision counts per (simulation, game, round, category)")
    print(f"   Sample rows:\n{decision_counts.head(20)}")

    # Pivot to get decision types as columns
    decision_pivot_per_sim = decision_counts.pivot_table(
        index=['simulation_id', 'game_number', 'round'],
        columns='decision_category',
        values='count',
        fill_value=0
    ).reset_index()
    print(f"\n   STEP 2: Pivoted decision counts (decision types as columns)")
    print(f"   Sample rows:\n{decision_pivot_per_sim.head(20)}")

    # Create complete grid for ALL simulations, games, and rounds
    # This ensures we average by total games, not just games that reached each round
    num_simulations = len(all_sim_data)
    print(f"\n   STEP 3: Creating complete grid for {num_simulations} simulations × {num_games} games × {max_rounds_per_game} rounds")
    complete_grid_per_sim = []
    for sim_id in range(num_simulations):
        for game_num in range(1, num_games + 1):
            for round_num in range(1, max_rounds_per_game + 1):
                complete_grid_per_sim.append({
                    'simulation_id': sim_id,
                    'game_number': game_num,
                    'round': round_num
                })

    complete_df_per_sim = pd.DataFrame(complete_grid_per_sim)
    print(f"   Total grid rows: {len(complete_df_per_sim)}")

    # Decision columns
    decision_columns = ['pass', 'verify_dummy', 'verify_other', 'volunteer_with_dummy', 'volunteer_correct']

    # Get columns that actually exist in the data
    available_columns = [col for col in decision_columns if col in decision_pivot_per_sim.columns]

    # Merge actual data with complete grid (fills missing with NaN, which we'll convert to 0)
    decision_pivot_per_sim_complete = complete_df_per_sim.merge(
        decision_pivot_per_sim,
        on=['simulation_id', 'game_number', 'round'],
        how='left'
    )
    print(f"\n   STEP 4: Merged with complete grid")
    print(f"   Sample rows before filling NaN:\n{decision_pivot_per_sim_complete.head(20)}")

    # Fill NaN values with 0 for decision categories
    for col in available_columns:
        if col in decision_pivot_per_sim_complete.columns:
            decision_pivot_per_sim_complete[col] = decision_pivot_per_sim_complete[col].fillna(0)
    print(f"\n   STEP 5: After filling NaN with 0")
    print(f"   Sample rows:\n{decision_pivot_per_sim_complete.head(20)}")

    # Now average across simulations by game_number and round
    # Sum across simulations, then divide by number of simulations
    groupby_cols = ['game_number', 'round']
    decision_pivot = decision_pivot_per_sim_complete.groupby(groupby_cols)[available_columns].sum().reset_index()
    print(f"\n   STEP 6: Summed across {num_simulations} simulations by (game_number, round)")
    print(f"   Sample rows (before dividing by num_simulations):\n{decision_pivot.head(20)}")

    # Divide by number of simulations to get the average
    for col in available_columns:
        decision_pivot[col] = decision_pivot[col] / num_simulations
    print(f"\n   STEP 7: After dividing by {num_simulations} simulations")
    print(f"   Sample rows (final averaged values):\n{decision_pivot.head(20)}")

    # Sort to ensure proper order
    decision_pivot = decision_pivot.sort_values(['game_number', 'round'])

    # Create game-round labels
    decision_pivot['game_round_label'] = (
        'G' + decision_pivot['game_number'].astype(str) +
        'R' + decision_pivot['round'].astype(str)
    )

    print(f"\n   STEP 8: Final decision pivot (sorted)")
    print(f"   Total games: {num_games}")
    print(f"   Total game-round slots: {len(decision_pivot)} ({num_games} games × {max_rounds_per_game} rounds)")
    print(f"   Final data:\n{decision_pivot}")

    # Prepare data arrays
    x_labels = decision_pivot['game_round_label'].tolist()
    x_pos = np.arange(len(x_labels))

    # Extract decision counts
    pass_counts = decision_pivot.get('pass', pd.Series([0]*len(decision_pivot))).values

    if has_dummy:
        # 5 categories for dummy simulations
        verify_dummy_counts = decision_pivot.get('verify_dummy', pd.Series([0]*len(decision_pivot))).values
        verify_other_counts = decision_pivot.get('verify_other', pd.Series([0]*len(decision_pivot))).values
        volunteer_with_dummy_counts = decision_pivot.get('volunteer_with_dummy', pd.Series([0]*len(decision_pivot))).values
        volunteer_correct_counts = decision_pivot.get('volunteer_correct', pd.Series([0]*len(decision_pivot))).values
    else:
        # 3 categories for no-dummy simulations
        verify_counts = (
            decision_pivot.get('verify_dummy', pd.Series([0]*len(decision_pivot))).values +
            decision_pivot.get('verify_other', pd.Series([0]*len(decision_pivot))).values
        )
        volunteer_counts = (
            decision_pivot.get('volunteer_with_dummy', pd.Series([0]*len(decision_pivot))).values +
            decision_pivot.get('volunteer_correct', pd.Series([0]*len(decision_pivot))).values
        )

    # Create the plot
    fig, ax = plt.subplots(figsize=(18, 6))

    # Identify games with wrong dummy answers
    wrong_answer_games = set()
    if dummy_correctness:
        for dummy_id, correctness_list in dummy_correctness.items():
            for game_num in range(1, num_games + 1):
                game_index = game_num - 1  # 0-indexed for correctness_list
                # Check if this game index is within correctness_list and is False
                if game_index < len(correctness_list) and not correctness_list[game_index]:
                    wrong_answer_games.add(game_num)

        # Add gray shading for wrong answer games
        if wrong_answer_games:
            y_max = 10  # Set a high enough value for background
            for game_num in sorted(wrong_answer_games):
                # Calculate bar positions for this game
                # Each game takes max_rounds_per_game bars
                x_start = (game_num - 1) * max_rounds_per_game - 0.5
                x_end = game_num * max_rounds_per_game - 0.5
                ax.axvspan(x_start, x_end, alpha=0.45, color='gray', zorder=0)

    # Create stacked bars with conditional coloring
    width = 0.8

    if has_dummy:
        # If there's at least one wrong answer game, show detailed differentiation for ALL games
        if wrong_answer_games:
            # Use 5 categories with detailed color coding for all games
            # Stack from bottom to top
            p1 = ax.bar(x_pos, pass_counts, width, label='Pass', color='#93c5fd')

            bottom1 = pass_counts
            p2 = ax.bar(x_pos, verify_dummy_counts, width, bottom=bottom1,
                       label="Verify: Dummy's", color='#dc2626')

            bottom2 = bottom1 + verify_dummy_counts
            p3 = ax.bar(x_pos, verify_other_counts, width, bottom=bottom2,
                       label="Verify: Other's", color='#fca5a5')

            bottom3 = bottom2 + verify_other_counts
            p4 = ax.bar(x_pos, volunteer_with_dummy_counts, width, bottom=bottom3,
                       label="Volunteer: w/ Dummy's Answer", color='#16a34a', alpha=0.45)

            bottom4 = bottom3 + volunteer_with_dummy_counts
            p5 = ax.bar(x_pos, volunteer_correct_counts, width, bottom=bottom4,
                       label="Volunteer: w/o Dummy's Answer", color='#16a34a')
        else:
            # All games have correct answers: use simplified 3-category view
            # Combine verify categories
            verify_all = verify_dummy_counts + verify_other_counts
            # Combine volunteer categories
            volunteer_all = volunteer_with_dummy_counts + volunteer_correct_counts

            p1 = ax.bar(x_pos, pass_counts, width, label='Pass', color='#93c5fd')
            p2 = ax.bar(x_pos, verify_all, width, bottom=pass_counts,
                       label='Verify', color='#fca5a5')
            p3 = ax.bar(x_pos, volunteer_all, width,
                       bottom=pass_counts + verify_all,
                       label='Volunteer', color='#86efac')
    else:
        # 3 categories with simple color coding
        # Stack from bottom to top: pass, verify, volunteer
        p1 = ax.bar(x_pos, pass_counts, width, label='Pass', color='#93c5fd')
        p2 = ax.bar(x_pos, verify_counts, width, bottom=pass_counts,
                   label='Verify', color='#fca5a5')
        p3 = ax.bar(x_pos, volunteer_counts, width,
                   bottom=pass_counts + verify_counts,
                   label='Volunteer', color='#86efac')

    # Customize the plot
    ax.set_xlabel('Game', fontsize=20, fontweight='bold')
    ax.set_ylabel('Average Decision Count', fontsize=20, fontweight='bold')

    # Create title with shortened prefix (only last 2 parts: -3 and -2)
    parts = prefix.split('_')
    if len(parts) >= 3:
        short_prefix = f"{parts[-3]}_{parts[-2]}"
    else:
        short_prefix = prefix
    title = f'Agent Decisions by Game & Round (Averaged)\n{short_prefix}'
    ax.set_title(title, fontsize=22, fontweight='bold', pad=60)

    # Set x-axis labels to show only game numbers
    # Calculate center position of each game
    game_tick_positions = [i * max_rounds_per_game + (max_rounds_per_game - 1) / 2 for i in range(num_games)]
    game_labels = [f'G{i+1}' for i in range(num_games)]
    ax.set_xticks(game_tick_positions)
    ax.set_xticklabels(game_labels, fontsize=16)

    # Add vertical lines to mark game boundaries (if multiple games)
    if num_games > 1:
        # Each game takes exactly max_rounds_per_game bars
        for i in range(1, num_games):
            boundary_pos = i * max_rounds_per_game - 0.5
            ax.axvline(x=boundary_pos, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    # Add legend above the plot and below the title
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), framealpha=0.9, ncol=5, fontsize=16)

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis limits and integer ticks
    ax.set_ylim([0, 3.5])
    ax.set_yticks(range(0, 4))
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'decisions_aggregated_{prefix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Plot saved to: {output_file}")

    # Save the data as CSV
    csv_file = output_dir / f'decisions_aggregated_{prefix}.csv'
    decision_pivot.to_csv(csv_file, index=False)
    print(f"   📊 Data saved to: {csv_file}")

    plt.close()

    # Calculate average total coins across simulations
    if all_coin_data:
        combined_coin_df = pd.concat(all_coin_data, ignore_index=True)

        # For rough_start, filter to only first 10 games
        if 'rough_start' in prefix:
            combined_coin_df = combined_coin_df[combined_coin_df['game_number'] <= 10].copy()
            decision_pivot = decision_pivot[decision_pivot['game_number'] <= 10].copy()

        # Average total coins and std across simulations by game_number (for line plot)
        stats_df = combined_coin_df.groupby('game_number')['total_coins'].agg(['mean', 'std']).reset_index()
        stats_df.columns = ['game_number', 'total_coins', 'std']
        # Fill NaN std with 0 (happens when only 1 simulation)
        stats_df['std'] = stats_df['std'].fillna(0)
        # Return stats data (for line plot), per-simulation data (for bar plot), decision data, and metadata
        return stats_df, combined_coin_df, decision_pivot, dummy_correctness, num_games
    else:
        return None, None, None, None, None


def plot_average_coins_line(coin_data_by_prefix: dict, dummy_info_by_prefix: dict, output_dir: Path):
    """
    Create a line plot showing average total coins across games.

    Args:
        coin_data_by_prefix: Dict mapping prefix names to DataFrames with game_number and total_coins
        dummy_info_by_prefix: Dict mapping prefix names to (dummy_correctness, num_games) tuples
        output_dir: Directory to save output plot
    """
    print(f"\n{'='*80}")
    print("Creating line plot: Average Total Coins per Game")
    print(f"{'='*80}")

    fig, ax = plt.subplots(figsize=(14, 8))

    # Add gray background for games where dummy gave wrong answer
    # Collect all wrong answer games across all prefixes
    wrong_answer_games = set()
    for prefix, (dummy_correctness, num_games) in dummy_info_by_prefix.items():
        if dummy_correctness:
            for dummy_id, correctness_list in dummy_correctness.items():
                for game_num in range(1, num_games + 1):
                    game_index = game_num - 1  # 0-indexed for correctness_list
                    # Check if this game index is within correctness_list and is False
                    if game_index < len(correctness_list) and not correctness_list[game_index]:
                        wrong_answer_games.add(game_num)

    # Add gray shading for wrong answer games
    if wrong_answer_games:
        y_max = 20  # Set a high enough value for background
        for game_num in sorted(wrong_answer_games):
            ax.axvspan(game_num - 0.5, game_num + 0.5, alpha=0.45, color='gray', zorder=0)

    # Define colors for different conditions
    colors = ['#2563eb', '#dc2626', '#16a34a', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']

    # Determine prefix type for title (trust_converge or rough_start)
    prefix_type = None
    if coin_data_by_prefix:
        first_prefix = next(iter(coin_data_by_prefix.keys()))
        if 'trust_converge' in first_prefix:
            prefix_type = 'trust_converge'
        elif 'rough_start' in first_prefix:
            prefix_type = 'rough_start'

    # Plot each prefix as a separate line with uncertainty bands
    for idx, (prefix, coin_df) in enumerate(coin_data_by_prefix.items()):
        color = colors[idx % len(colors)]
        # Extract items at index -3 and -2 from prefix for legend label
        parts = prefix.split('_')
        if len(parts) >= 3:
            legend_label = f"{parts[-3]}_{parts[-2]}"
        else:
            legend_label = prefix

        # Plot mean line
        ax.plot(
            coin_df['game_number'],
            coin_df['total_coins'],
            marker='o',
            linewidth=3,
            markersize=8,
            label=legend_label,
            color=color
        )

        # Add uncertainty band (±1 standard deviation)
        if 'std' in coin_df.columns:
            lower_bound = coin_df['total_coins'] - coin_df['std']
            upper_bound = coin_df['total_coins'] + coin_df['std']
            ax.fill_between(
                coin_df['game_number'],
                lower_bound,
                upper_bound,
                color=color,
                alpha=0.2,
                linewidth=0
            )

    # Customize the plot
    ax.set_xlabel('Game Number', fontsize=20, fontweight='bold')
    ax.set_ylabel('Total Coins earned per Game', fontsize=20, fontweight='bold')

    # Create title with prefix type on first line
    if prefix_type == 'rough_start':
        title = f' Trust recovery from a rough start \nAverage End-of-Game Total Coins Across Games'
    elif prefix_type == 'trust_converge':
        title = f' Trust convergence from a smooth start \nAverage End-of-Game Total Coins Across Games'
    else:
        title = 'Average End-of-Game Total Coins Across Games'
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)

    # Add legend at top, below title, 2 columns x 2 rows
    ax.legend(loc='right', bbox_to_anchor=(1.4, 0.5), framealpha=0.9,
              ncol=1, fontsize=20, edgecolor='#cccccc', fancybox=True)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add grid for better readability
    ax.grid(axis='both', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Set y-axis limits
    ax.set_ylim([-1, 17])

    # Set x-axis to show integer game numbers
    if coin_data_by_prefix:
        first_df = next(iter(coin_data_by_prefix.values()))
        max_game = int(first_df['game_number'].max())
        ax.set_xticks(range(1, max_game + 1, max(1, max_game // 20)))  # Show ~20 ticks

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    # Use prefix_type in filename so rough_start and trust_converge don't overwrite each other
    suffix = f'_{prefix_type}' if prefix_type else ''
    output_file = output_dir / f'average_coins_line_plot{suffix}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Line plot saved to: {output_file}")

    # Save the data as CSV - combine all prefixes into one dataframe
    combined_coin_data = []
    for prefix, coin_df in coin_data_by_prefix.items():
        temp_df = coin_df.copy()
        temp_df['condition'] = prefix
        combined_coin_data.append(temp_df)

    if combined_coin_data:
        combined_coin_df = pd.concat(combined_coin_data, ignore_index=True)
        # Reorder columns for clarity
        combined_coin_df = combined_coin_df[['condition', 'game_number', 'total_coins']]
        csv_file = output_dir / 'average_coins_line_plot.csv'
        combined_coin_df.to_csv(csv_file, index=False)
        print(f"   📊 Data saved to: {csv_file}")

    plt.close()


def plot_total_coins_bar(per_sim_coin_data_by_prefix: dict, dummy_info_by_prefix: dict, output_dir: Path):
    """
    Create a bar plot showing average total coins across simulations for each prefix.
    Only counts coins from games where dummy agents gave correct answers.

    For each prefix:
      1. For each simulation: filter correct answer games, sum total coins
      2. Average those sums across simulations

    Args:
        per_sim_coin_data_by_prefix: Dict mapping prefix names to DataFrames with game_number, total_coins, simulation_id
        dummy_info_by_prefix: Dict mapping prefix names to (dummy_correctness, num_games) tuples
        output_dir: Directory to save output plot
    """
    print(f"\n{'='*80}")
    print("Creating bar plot: Average Total Coins Across Games (Correct Answer Games Only)")
    print(f"{'='*80}")

    # Calculate average total coins for each prefix
    totals = []
    prefixes = []
    for prefix, per_sim_coin_df in per_sim_coin_data_by_prefix.items():
        # Get dummy correctness info for this prefix
        dummy_correctness, num_games = dummy_info_by_prefix.get(prefix, ({}, 0))

        # Step 1: Find games where ALL dummies gave correct answers
        correct_answer_games = set()
        if dummy_correctness:
            for game_num in range(1, num_games + 1):
                game_index = game_num - 1
                all_correct = True

                # Check if all dummies gave correct answers for this game
                for dummy_id, correctness_list in dummy_correctness.items():
                    if game_index < len(correctness_list) and not correctness_list[game_index]:
                        all_correct = False
                        break

                if all_correct:
                    correct_answer_games.add(game_num)
        else:
            # No dummy agents, include all games
            correct_answer_games = set(per_sim_coin_df['game_number'].unique())

        # Step 2: For each simulation, filter to correct answer games and sum
        per_simulation_sums = []
        for sim_id in per_sim_coin_df['simulation_id'].unique():
            sim_df = per_sim_coin_df[per_sim_coin_df['simulation_id'] == sim_id]
            filtered_sim_df = sim_df[sim_df['game_number'].isin(correct_answer_games)]
            sim_total = filtered_sim_df['total_coins'].sum()
            per_simulation_sums.append(sim_total)

        # Step 3: Average across simulations
        avg_total_coins = np.mean(per_simulation_sums) if per_simulation_sums else 0

        num_sims = len(per_simulation_sums)
        num_correct_games = len(correct_answer_games)
        print(f"   {prefix}: {num_correct_games}/{num_games} games with correct answers, "
              f"n={num_sims} sims, avg_total={avg_total_coins:.2f}")

        totals.append(avg_total_coins)
        prefixes.append(prefix)

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define colors
    colors = ['#2563eb', '#dc2626', '#16a34a', '#f59e0b', '#8b5cf6', '#ec4899', '#06b6d4', '#84cc16']
    bar_colors = [colors[i % len(colors)] for i in range(len(prefixes))]

    # Create bars
    x_pos = np.arange(len(prefixes))
    bars = ax.bar(x_pos, totals, color=bar_colors, width=0.6)

    # Customize the plot
    ax.set_xlabel('Simulation Condition', fontsize=20, fontweight='bold')
    ax.set_ylabel('Average Total Coins (Across Simulations)', fontsize=20, fontweight='bold')
    ax.set_title('Average Total Coins Earned (Correct Answer Games) by Condition', fontsize=22, fontweight='bold', pad=20)

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(prefixes, rotation=45, ha='right', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=16)

    # Add value labels on top of bars
    for i, (bar, total) in enumerate(zip(bars, totals)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{total:.1f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'total_coins_bar_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n   ✅ Bar plot saved to: {output_file}")

    # Save the data as CSV
    bar_data_df = pd.DataFrame({
        'condition': prefixes,
        'average_total_coins': totals
    })
    csv_file = output_dir / 'total_coins_bar_plot.csv'
    bar_data_df.to_csv(csv_file, index=False)
    print(f"   📊 Data saved to: {csv_file}")

    plt.close()


def main():
    """Main entry point."""
    # =========================================================================
    # CONFIGURATION: Two condition groups — both are processed in a single run
    # =========================================================================
    CONDITION_GROUPS = {
        'rough_start': [
            'rough_start_high_reflection_max',
            'rough_start_high_none_max',
            'rough_start_none_reflection_max',
            'rough_start_none_none_max',
        ],
        'trust_converge': [
            'trust_converge_high_reflection_max',
            'trust_converge_high_none_max',
            'trust_converge_none_reflection_max',
            'trust_converge_none_none_max',
        ],
    }
    # =========================================================================

    results_dir = Path('results')
    output_dir = Path('analysis_output')

    print("=" * 80)
    print("AGGREGATE ANALYSIS: Multiple Simulations")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")

    all_decision_data = {}

    for group_name, prefixes in CONDITION_GROUPS.items():
        print(f"\n{'='*80}")
        print(f"Processing condition group: {group_name}")
        print(f"Prefixes: {prefixes}")
        print(f"{'='*80}")

        coin_data_by_prefix = {}
        per_sim_coin_data_by_prefix = {}
        decision_data_by_prefix = {}
        dummy_info_by_prefix = {}

        for prefix in prefixes:
            avg_coin_df, per_sim_coin_df, decision_df, dummy_correctness, num_games = aggregate_and_plot(results_dir, prefix, output_dir)
            if avg_coin_df is not None:
                coin_data_by_prefix[prefix] = avg_coin_df
                per_sim_coin_data_by_prefix[prefix] = per_sim_coin_df
                decision_data_by_prefix[prefix] = decision_df
                dummy_info_by_prefix[prefix] = (dummy_correctness, num_games)

        # Create line plot and bar plot for this condition group
        if coin_data_by_prefix:
            plot_average_coins_line(coin_data_by_prefix, dummy_info_by_prefix, output_dir)
            plot_total_coins_bar(per_sim_coin_data_by_prefix, dummy_info_by_prefix, output_dir)

        all_decision_data.update(decision_data_by_prefix)

    # Combine all decision CSVs into one master CSV
    if all_decision_data:
        print(f"\n{'='*80}")
        print("Creating combined decision data CSV")
        print(f"{'='*80}")

        combined_decision_data = []
        for prefix, decision_df in all_decision_data.items():
            temp_df = decision_df.copy()
            temp_df['condition'] = prefix
            combined_decision_data.append(temp_df)

        combined_decision_df = pd.concat(combined_decision_data, ignore_index=True)
        cols = ['condition'] + [col for col in combined_decision_df.columns if col != 'condition']
        combined_decision_df = combined_decision_df[cols]

        csv_file = output_dir / 'decisions_aggregated_all_conditions.csv'
        combined_decision_df.to_csv(csv_file, index=False)
        print(f"   📊 Combined decision data saved to: {csv_file}")

    total_prefixes = sum(len(p) for p in CONDITION_GROUPS.values())
    print("\n" + "=" * 80)
    print(f"✅ Analysis complete! Generated:")
    print(f"   - {total_prefixes} stacked bar plot(s) + individual CSVs")
    print(f"   - {len(CONDITION_GROUPS)} line plot(s): average_coins_line_plot_rough_start.png, average_coins_line_plot_trust_converge.png")
    print(f"   - 1 combined decision CSV (all conditions)")
    print(f"📁 All files saved to: {output_dir}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

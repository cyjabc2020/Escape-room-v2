# -*- coding: utf-8 -*-
"""
Calculate mean and standard deviation of total coins earned in a specific game across simulations.

This script takes one or more folder prefixes and a game order index, finds all matching result folders,
and calculates statistics on total coins earned by all players in that game.

Usage:
    python get_coin_variance.py <game_idx> <prefix1> [prefix2] [prefix3] ...

Example:
    python get_coin_variance.py 10 rough_start_high_none_max trust_converge_high_none_max
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


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
        print(f"❌ Results directory not found: {results_dir}")
        return matching_folders

    for folder in results_dir.iterdir():
        if folder.is_dir() and folder.name.startswith(prefix):
            matching_folders.append(folder)

    return sorted(matching_folders)


def get_game_total_coins(folder_path: Path, game_idx: int):
    """
    Get total coins earned by all players at the end of a specific game.

    Args:
        folder_path: Path to simulation folder
        game_idx: Game index (1-indexed, chronological order)

    Returns:
        Total coins for that game, or None if not found
    """
    # Read decisions CSV to get game IDs in chronological order
    decisions_csv = folder_path / "game_data_agent_decisions.csv"

    if not decisions_csv.exists():
        print(f"   Warning: Agent decisions CSV not found in {folder_path.name}")
        return None

    decisions_df = pd.read_csv(decisions_csv)
    unique_game_ids = sorted(decisions_df['game_id'].unique())

    # Check if game_idx is valid
    if game_idx < 1 or game_idx > len(unique_game_ids):
        print(f"   Warning: Game index {game_idx} out of range (1-{len(unique_game_ids)}) in {folder_path.name}")
        return None

    # Get the game_id for the requested game index
    target_game_id = unique_game_ids[game_idx - 1]

    # Get all rounds for this game
    game_decisions = decisions_df[decisions_df['game_id'] == target_game_id].copy()

    if len(game_decisions) == 0:
        print(f"   Warning: No decisions found for game {target_game_id} in {folder_path.name}")
        return None

    # Get the final round (highest round number) for this game
    final_round = game_decisions['round'].max()
    final_round_data = game_decisions[game_decisions['round'] == final_round]

    # Sum up coins_end_of_round for all agents in the final round
    if 'coins_end_of_round' not in final_round_data.columns:
        print(f"   Warning: 'coins_end_of_round' column not found in {folder_path.name}")
        return None

    total_coins = final_round_data['coins_end_of_round'].sum()

    return total_coins


def process_prefix(prefix: str, game_idx: int, results_dir: Path, output_dir: Path):
    """
    Process a single prefix and return statistics.

    Args:
        prefix: Folder prefix to match
        game_idx: Game index (1-indexed)
        results_dir: Path to results directory
        output_dir: Path to output directory

    Returns:
        Dictionary with statistics, or None if no data found
    """
    print("\n" + "=" * 80)
    print(f"Processing prefix: {prefix}")
    print("=" * 80)

    # Find matching folders
    matching_folders = find_matching_folders(results_dir, prefix)

    if not matching_folders:
        print(f"❌ No matching folders found for prefix: {prefix}")
        return None

    print(f"Found {len(matching_folders)} matching simulation(s):")
    for folder in matching_folders:
        print(f"  - {folder.name}")
    print()

    # Collect total coins from all matching simulations
    all_total_coins = []
    for folder in matching_folders:
        print(f"Processing: {folder.name}")
        total_coins = get_game_total_coins(folder, game_idx)
        if total_coins is not None:
            all_total_coins.append({
                'simulation_folder': folder.name,
                'total_coins': total_coins
            })
            print(f"   ✅ Total coins: {total_coins}")
        else:
            print(f"   ⚠️  Could not retrieve coin data")

    if not all_total_coins:
        print(f"❌ No coin data found for prefix: {prefix}")
        return None

    # Create DataFrame
    coins_df = pd.DataFrame(all_total_coins)
    coins_df['prefix'] = prefix

    # Calculate statistics
    mean_coins = coins_df['total_coins'].mean()
    std_coins = coins_df['total_coins'].std()
    min_coins = coins_df['total_coins'].min()
    max_coins = coins_df['total_coins'].max()

    print("\n" + "-" * 80)
    print(f"STATISTICS for {prefix}")
    print("-" * 80)
    print(f"Number of simulations: {len(coins_df)}")
    print(f"Mean total coins: {mean_coins:.2f}")
    print(f"Standard deviation: {std_coins:.2f}")
    print(f"Min total coins: {min_coins}")
    print(f"Max total coins: {max_coins}")
    print("-" * 80)

    # Save individual prefix data to CSV
    output_file = output_dir / f'coin_variance_{prefix}_game{game_idx}.csv'
    coins_df.to_csv(output_file, index=False)
    print(f"📊 Data saved to: {output_file}")

    # Return statistics
    return {
        'prefix': prefix,
        'game_idx': game_idx,
        'n_simulations': len(coins_df),
        'mean': mean_coins,
        'std': std_coins,
        'min': min_coins,
        'max': max_coins
    }


def main():
    """Main entry point."""
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: python get_coin_variance.py <game_idx> <prefix1> [prefix2] [prefix3] ...")
        print("Example: python get_coin_variance.py 10 rough_start_high_none_max trust_converge_high_none_max")
        sys.exit(1)

    try:
        game_idx = int(sys.argv[1])
    except ValueError:
        print(f"❌ Error: game_idx must be an integer, got '{sys.argv[1]}'")
        sys.exit(1)

    prefixes = sys.argv[2:]

    results_dir = Path('results')
    output_dir = Path('coin_variance_output')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(f"COIN STATISTICS ANALYSIS")
    print("=" * 80)
    print(f"Game Index: {game_idx} (chronological order)")
    print(f"Number of prefixes: {len(prefixes)}")
    print(f"Prefixes: {', '.join(prefixes)}")
    print(f"Results directory: {results_dir}")
    print("=" * 80)

    # Process each prefix
    all_statistics = []
    for prefix in prefixes:
        stats = process_prefix(prefix, game_idx, results_dir, output_dir)
        if stats is not None:
            all_statistics.append(stats)

    if not all_statistics:
        print("\n❌ No data found for any prefix")
        sys.exit(0)

    # Create combined summary DataFrame
    summary_df = pd.DataFrame(all_statistics)

    # Save combined summary
    summary_file = output_dir / f'coin_variance_summary_all_prefixes_game{game_idx}.csv'
    summary_df.to_csv(summary_file, index=False)

    # Print combined summary
    print("\n\n" + "=" * 80)
    print(f"COMBINED SUMMARY - All Prefixes")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"\n📊 Combined summary saved to: {summary_file}")


if __name__ == "__main__":
    main()

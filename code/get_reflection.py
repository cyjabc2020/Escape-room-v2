# -*- coding: utf-8 -*-
"""
Extract reflections from a specific game across multiple simulation runs.

This script takes a folder prefix and game order index, finds all matching result folders,
and extracts reflections from that specific game across all simulations.

Usage:
    python get_reflection.py <prefix> <game_idx>

Example:
    python get_reflection.py trust_converge_high_none_max 5
"""

import pandas as pd
from pathlib import Path
import sys
import json


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


def get_game_reflections(folder_path: Path, game_idx: int):
    """
    Extract reflections from a specific game (by chronological order) in a simulation folder.

    Args:
        folder_path: Path to simulation folder
        game_idx: Game index (1-indexed, chronological order)

    Returns:
        DataFrame with reflections, or None if not found
    """
    # First, get the game_id from agent decisions CSV
    decisions_csv = folder_path / "game_data_agent_decisions.csv"
    reflections_csv = folder_path / "game_data_reflections.csv"

    if not decisions_csv.exists():
        print(f"   Warning: Agent decisions CSV not found in {folder_path.name}")
        return None

    if not reflections_csv.exists():
        print(f"   Warning: Reflections CSV not found in {folder_path.name}")
        return None

    # Read decisions CSV to get game IDs in chronological order
    decisions_df = pd.read_csv(decisions_csv)
    unique_game_ids = sorted(decisions_df['game_id'].unique())

    # Check if game_idx is valid
    if game_idx < 1 or game_idx > len(unique_game_ids):
        print(f"   Warning: Game index {game_idx} out of range (1-{len(unique_game_ids)}) in {folder_path.name}")
        return None

    # Get the game_id for the requested game index
    target_game_id = unique_game_ids[game_idx - 1]

    # Read reflections CSV
    reflections_df = pd.read_csv(reflections_csv)

    # Filter for the target game
    game_reflections = reflections_df[reflections_df['game_id'] == target_game_id].copy()

    if len(game_reflections) == 0:
        print(f"   Warning: No reflections found for game {target_game_id} in {folder_path.name}")
        return None

    # Filter out empty/NaN reflections
    game_reflections = game_reflections[game_reflections['reflection'].notna()].copy()
    game_reflections = game_reflections[game_reflections['reflection'] != ''].copy()

    # Get Round 1 decisions from decisions CSV
    round1_decisions = decisions_df[
        (decisions_df['game_id'] == target_game_id) &
        (decisions_df['round'] == 1)
    ][['agent_id', 'agent_decision']].copy()
    round1_decisions.columns = ['agent_id', 'round1_decision']

    # Merge Round 1 decisions with reflections
    game_reflections = game_reflections.merge(round1_decisions, on='agent_id', how='left')

    # Add simulation folder name
    game_reflections['simulation_folder'] = folder_path.name

    return game_reflections


def main():
    """Main entry point."""
    # Parse command line arguments
    if len(sys.argv) != 3:
        print("Usage: python get_reflection.py <prefix> <game_idx>")
        print("Example: python get_reflection.py trust_converge_high_none_max 5")
        sys.exit(1)

    prefix = sys.argv[1]
    try:
        game_idx = int(sys.argv[2])
    except ValueError:
        print(f"❌ Error: game_idx must be an integer, got '{sys.argv[2]}'")
        sys.exit(1)

    results_dir = Path('results')

    print("=" * 80)
    print(f"REFLECTION EXTRACTION")
    print("=" * 80)
    print(f"Prefix: {prefix}")
    print(f"Game Index: {game_idx} (chronological order)")
    print(f"Results directory: {results_dir}")
    print()

    # Find matching folders
    matching_folders = find_matching_folders(results_dir, prefix)

    if not matching_folders:
        print(f"❌ No matching folders found for prefix: {prefix}")
        sys.exit(1)

    print(f"Found {len(matching_folders)} matching simulation(s):")
    for folder in matching_folders:
        print(f"  - {folder.name}")
    print()

    # Collect reflections from all matching simulations
    all_reflections = []
    for folder in matching_folders:
        print(f"Processing: {folder.name}")
        reflections_df = get_game_reflections(folder, game_idx)
        if reflections_df is not None and len(reflections_df) > 0:
            all_reflections.append(reflections_df)
            print(f"   ✅ Found {len(reflections_df)} reflection(s)")
        else:
            print(f"   ⚠️  No reflections found")

    if not all_reflections:
        print("\n❌ No reflections found across any simulations")
        sys.exit(0)

    # Combine all reflections
    combined_reflections = pd.concat(all_reflections, ignore_index=True)

    # Sort by simulation folder, agent
    combined_reflections = combined_reflections.sort_values(
        ['simulation_folder', 'agent_id']
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"✅ Total reflections collected: {len(combined_reflections)}")
    print("=" * 80)

    # Save to CSV
    output_dir = Path('reflection_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'reflections_{prefix}_game{game_idx}.csv'
    combined_reflections.to_csv(output_file, index=False)
    print(f"\n📊 Reflections saved to: {output_file}")

    # Also print summary to console
    print("\n" + "=" * 80)
    print("REFLECTION SUMMARY")
    print("=" * 80)
    for idx, row in combined_reflections.iterrows():
        print(f"\n{'='*80}")
        print(f"Simulation: {row['simulation_folder']}")
        print(f"Game ID: {row['game_id']} | Agent: {row['agent_id']}")
        print(f"Round 1 Decision: {row.get('round1_decision', 'N/A')}")
        print(f"Survived: {row.get('survived', 'N/A')} | Was Winner: {row.get('was_winner', 'N/A')}")
        print(f"Game Outcome: {row.get('game_outcome', 'N/A')} | Total Rounds: {row.get('total_rounds', 'N/A')}")
        print(f"\nReflection:")
        print(f"{row['reflection']}")
        print("-" * 80)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
Extract reasoning from a specific game and round across multiple simulation runs.

This script takes a folder prefix, game order index, and round index, finds all matching
result folders, and extracts reasoning from that specific game and round across all simulations.

Usage:
    python get_reasoning.py <prefix> <game_idx> <round_idx>

Example:
    python get_reasoning.py trust_converge_high_none_max 5 1
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


def get_game_reasoning(folder_path: Path, game_idx: int, round_idx: int):
    """
    Extract reasoning from a specific game and round in a simulation folder.

    Args:
        folder_path: Path to simulation folder
        game_idx: Game index (1-indexed, chronological order)
        round_idx: Round index (1-indexed)

    Returns:
        DataFrame with reasoning, or None if not found
    """
    # Get the game_id from agent decisions CSV
    decisions_csv = folder_path / "game_data_agent_decisions.csv"

    if not decisions_csv.exists():
        print(f"   Warning: Agent decisions CSV not found in {folder_path.name}")
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

    # Find the JSON file for this game
    json_file = folder_path / f"game_{target_game_id}.json"

    if not json_file.exists():
        print(f"   Warning: Game JSON not found: {json_file.name}")
        return None

    # Read the JSON file
    try:
        with open(json_file, 'r') as f:
            game_data = json.load(f)
    except Exception as e:
        print(f"   Warning: Failed to read JSON {json_file.name}: {e}")
        return None

    # Navigate to the round data
    if 'rounds' not in game_data or not isinstance(game_data['rounds'], list):
        print(f"   Warning: No 'rounds' data in {json_file.name}")
        return None

    # Check if round_idx is valid
    if round_idx < 1 or round_idx > len(game_data['rounds']):
        print(f"   Warning: Round index {round_idx} out of range (1-{len(game_data['rounds'])}) in {json_file.name}")
        return None

    # Get the round data (rounds are 0-indexed in the array)
    round_data = game_data['rounds'][round_idx - 1]

    # Extract reasoning from round data
    if 'reasoning' not in round_data:
        print(f"   Warning: No 'reasoning' in round {round_idx} of {json_file.name}")
        return None

    reasoning_dict = round_data['reasoning']
    decisions_dict = round_data.get('decisions', {})

    reasoning_records = []
    for agent_id, reasoning_text in reasoning_dict.items():
        if reasoning_text:
            reasoning_records.append({
                'game_id': target_game_id,
                'round': round_idx,
                'agent_id': agent_id,
                'decision': decisions_dict.get(agent_id, 'N/A'),
                'reasoning': reasoning_text
            })

    if not reasoning_records:
        print(f"   Warning: No reasoning found in round {round_idx} of {json_file.name}")
        return None

    reasoning_df = pd.DataFrame(reasoning_records)
    reasoning_df['simulation_folder'] = folder_path.name

    return reasoning_df


def main():
    """Main entry point."""
    # Parse command line arguments
    if len(sys.argv) != 4:
        print("Usage: python get_reasoning.py <prefix> <game_idx> <round_idx>")
        print("Example: python get_reasoning.py trust_converge_high_none_max 5 1")
        sys.exit(1)

    prefix = sys.argv[1]
    try:
        game_idx = int(sys.argv[2])
        round_idx = int(sys.argv[3])
    except ValueError:
        print(f"❌ Error: game_idx and round_idx must be integers")
        sys.exit(1)

    results_dir = Path('results')

    print("=" * 80)
    print(f"REASONING EXTRACTION")
    print("=" * 80)
    print(f"Prefix: {prefix}")
    print(f"Game Index: {game_idx} (chronological order)")
    print(f"Round Index: {round_idx}")
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

    # Collect reasoning from all matching simulations
    all_reasoning = []
    for folder in matching_folders:
        print(f"Processing: {folder.name}")
        reasoning_df = get_game_reasoning(folder, game_idx, round_idx)
        if reasoning_df is not None and len(reasoning_df) > 0:
            all_reasoning.append(reasoning_df)
            print(f"   ✅ Found {len(reasoning_df)} reasoning record(s)")
        else:
            print(f"   ⚠️  No reasoning found")

    if not all_reasoning:
        print("\n❌ No reasoning found across any simulations")
        sys.exit(0)

    # Combine all reasoning
    combined_reasoning = pd.concat(all_reasoning, ignore_index=True)

    # Sort by simulation folder, agent
    combined_reasoning = combined_reasoning.sort_values(
        ['simulation_folder', 'agent_id']
    ).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f"✅ Total reasoning records collected: {len(combined_reasoning)}")
    print("=" * 80)

    # Save to CSV
    output_dir = Path('reasoning_output')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f'reasoning_{prefix}_game{game_idx}_round{round_idx}.csv'
    combined_reasoning.to_csv(output_file, index=False)
    print(f"\n📊 Reasoning saved to: {output_file}")

    # Also print summary to console
    print("\n" + "=" * 80)
    print("REASONING SUMMARY")
    print("=" * 80)
    for idx, row in combined_reasoning.iterrows():
        print(f"\n{'='*80}")
        print(f"Simulation: {row['simulation_folder']}")
        print(f"Game ID: {row['game_id']} | Round: {row['round']} | Agent: {row['agent_id']}")
        print(f"Decision: {row['decision']}")
        print(f"\nReasoning:")
        print(f"{row['reasoning']}")
        print("-" * 80)


if __name__ == "__main__":
    main()

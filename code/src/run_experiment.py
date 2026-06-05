"""
Master experiment runner for running multiple games with master settings
"""
import sys
from pathlib import Path
from games.escape_room import EscapeRoomGame
from utils.master_settings import MasterSettings
import pandas as pd


def generate_score_summaries(folder_path: Path) -> tuple:
    """
    Generate simulation scores and final scores from game data.

    Args:
        folder_path: Path to experiment folder containing game_data_agent_decisions.csv

    Returns:
        Tuple of (simulation_df, final_df)
    """
    csv_file = folder_path / "game_data_agent_decisions.csv"

    if not csv_file.exists():
        print(f"\n⚠️  Warning: Could not find {csv_file}, skipping score summary")
        return None, None

    print(f"\n📊 Generating score summaries...")

    # Read the CSV
    df = pd.read_csv(csv_file)

    # Verify required columns exist
    required_cols = ['game_id', 'round', 'agent_id', 'agent_status_end_of_round', 'coins_end_of_round']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"⚠️  Warning: Missing required columns: {missing_cols}, skipping score summary")
        return None, None

    # For each game, find the last round
    last_rounds = df.groupby('game_id')['round'].max().reset_index()
    last_rounds.columns = ['game_id', 'last_round']

    # Merge to get only last round data
    df_with_last = df.merge(last_rounds, on='game_id')
    df_last_round = df_with_last[df_with_last['round'] == df_with_last['last_round']]

    # Extract final results
    simulation_scores = df_last_round[['agent_id', 'game_id', 'agent_status_end_of_round', 'coins_end_of_round']].copy()
    simulation_scores.columns = ['player', 'game_id', 'survival', 'coins']

    # Convert survival from "ALIVE"/"DIED" to True/False
    simulation_scores['survival'] = simulation_scores['survival'] == 'ALIVE'

    # Sort by game_id and player
    simulation_scores = simulation_scores.sort_values(['game_id', 'player']).reset_index(drop=True)

    # Save simulation scores
    sim_output = folder_path / "simulation_scores.csv"
    simulation_scores.to_csv(sim_output, index=False)

    # Calculate final scores per player
    final_scores = simulation_scores.groupby('player').agg({
        'survival': ['mean', 'count'],
        'coins': 'mean'
    }).reset_index()

    # Flatten column names
    final_scores.columns = ['player', 'avg_survival_pct', 'num_games', 'avg_coins']

    # Convert survival rate to percentage
    final_scores['avg_survival_pct'] = final_scores['avg_survival_pct'] * 100

    # Reorder columns
    final_scores = final_scores[['player', 'avg_survival_pct', 'avg_coins', 'num_games']]

    # Sort by player name
    final_scores = final_scores.sort_values('player').reset_index(drop=True)

    # Save final scores
    final_output = folder_path / "final_scores.csv"
    final_scores.to_csv(final_output, index=False)

    # Print final scores table
    print("\n" + "=" * 80)
    print("🏆 FINAL SCORES (Aggregated Across All Games)")
    print("=" * 80)
    print(f"\n{'Player':<10} {'Survival %':<15} {'Avg Coins':<15} {'Games Played':<15}")
    print("-" * 80)
    for _, row in final_scores.iterrows():
        print(f"{row['player']:<10} {row['avg_survival_pct']:>6.1f}%{'':<8} {row['avg_coins']:>6.2f}{'':<8} {int(row['num_games']):>6}")
    print("=" * 80)

    # Overall statistics
    overall_survival = simulation_scores['survival'].mean() * 100
    overall_coins = simulation_scores['coins'].mean()
    print(f"\n🌟 Overall Statistics:")
    print(f"   Average survival rate: {overall_survival:.1f}%")
    print(f"   Average coins per player-game: {overall_coins:.2f}")

    return simulation_scores, final_scores


def run_experiment(settings_file: str = None):
    """
    Run an experiment based on master settings, with optional warm-start from checkpoint

    Args:
        settings_file: Path to JSON master settings file (optional)
    """
    # Normal start (no checkpoint)
    settings = MasterSettings(settings_file)
    starting_game_number = 1
    persistent_agents = None
    used_questions_set = set()

    # Create fresh logger with experiment-specific folder
    from utils.game_logger import GameLogger
    shared_logger = GameLogger(
        output_dir=settings.get_log_directory(),
        experiment_name=settings.get_experiment_name(),
        settings=settings.config
    )

    # Print experiment summary
    settings.print_summary()

    # Extract settings
    num_players = settings.get_num_players()
    num_games = settings.get_num_games()
    max_rounds = settings.get_max_rounds_per_game()
    player_models = settings.get_player_models()
    memory_config = settings.get_memory_config()
    # Extract dummy_config from memory_config if nested, otherwise try top level
    dummy_config = memory_config.get("dummy_config", settings.config.get("dummy_config", {}))
    initial_coins_config = settings.config.get("initial_coins_config", {})
    reasoning_effort_config = settings.config.get("reasoning_effort_config", {})

    print(f"\n🚀 Starting experiment: {settings.get_experiment_name()}")
    print(f"Running {num_games} game(s) with {num_players} players each...\n")

    # Create shared question bank to prevent duplicate questions across games
    from utils.question_bank import QuestionBank

    shared_question_bank = QuestionBank()

    question_sets = settings.get_question_sets()

    print(f"   Questions needed: {num_players * num_games} ({num_players} per game × {num_games} games)\n")

    # Run multiple games (starting from correct position for warm-start)
    for game_num in range(starting_game_number, num_games + 1):
        print("\n" + "=" * 80)
        print(f"🎲 GAME {game_num}/{num_games}")
        print("=" * 80)

        game_question_sets = question_sets

        # Create and setup game with shared logger
        game = EscapeRoomGame(shared_logger=shared_logger)

        try:
            game.setup_game(
                num_players=num_players,
                question_sets=game_question_sets,
                agent_models=player_models,
                memory_config=memory_config,
                dummy_config=dummy_config,
                num_games=num_games,
                persistent_agents=persistent_agents,
                shared_question_bank=shared_question_bank,
                initial_coins_config=initial_coins_config,
                reasoning_effort_config=reasoning_effort_config
            )

            # Run the game
            game.run_game(max_rounds=max_rounds, game_number=game_num)

            # Save agents for next game if any have persist_across_games=true
            for agent_id, config in memory_config.items():
                if isinstance(config, dict) and config.get("persist_across_games", False):
                    if persistent_agents is None:
                        persistent_agents = {}
                    # Store the agent for next game
                    if agent_id in game.agents:
                        persistent_agents[agent_id] = game.agents[agent_id]
                        print(f"\n💾 Saving {agent_id}'s memory for next game ({len(game.agents[agent_id].memory)} memories)")

            print(f"\n✅ Game {game_num}/{num_games} completed!")

            # Wait between games to avoid rate limiting (but not after the last game)
            if game_num < num_games:
                wait_time = settings.get_wait_between_games()
                if wait_time > 0:
                    print(f"\n⏳ Waiting {wait_time} seconds before next game to avoid rate limiting...")
                    import time
                    time.sleep(wait_time)

        except Exception as e:
            print(f"\n❌ Error in game {game_num}: {e}")
            if settings.should_print_progress():
                import traceback
                traceback.print_exc()
            continue

    # Final summary
    print("\n" + "=" * 80)
    print(f"🏁 EXPERIMENT COMPLETE: {settings.get_experiment_name()}")
    print("=" * 80)
    print(f"Total games run: {num_games}")

    # Save all games to CSV files
    print(f"\n💾 Saving all {num_games} games to CSV files...")
    shared_logger.save_to_csv()

    # Generate score summaries
    simulation_df, final_df = generate_score_summaries(Path(shared_logger.output_dir))

    print(f"\n📁 All experiment data saved to: {shared_logger.output_dir}/")
    print("\n📊 Files generated:")
    print(f"  📄 experiment_settings.json - Full experiment configuration")
    print(f"  📊 game_data_agent_decisions.csv - All rounds from all games")
    print(f"  📊 game_data_capsule_status.csv - All capsules from all games")
    print(f"  📊 game_data_summary.csv - Summary of all games")
    print(f"  📊 game_data_reflections.csv - Post-game reflections from all players")
    if simulation_df is not None:
        print(f"  📊 simulation_scores.csv - Per-game survival and coin results")
        print(f"  📊 final_scores.csv - Aggregated player performance metrics")
    print(f"  📄 game_*.json - Individual game data files")
    print("=" * 80)


def main():
    """Main entry point

    Usage:
        # Normal start
        python run_experiment.py <settings_file>

        # Resume from checkpoint
        python run_experiment.py --resume <checkpoint_file> [--settings <new_settings_file>]

        # List available checkpoints
        python run_experiment.py --list-checkpoints [experiment_name]
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run multi-agent game experiments with warm-start support')
    parser.add_argument('settings_file', nargs='?', help='Path to settings JSON file')
    parser.add_argument('--settings', dest='new_settings',
                       help='New settings file to merge with checkpoint (optional)')

    args = parser.parse_args()


    # Normal start
    if args.settings_file:
        settings_file = args.settings_file
        print(f"Loading settings from: {settings_file}")
    else:
        # Check for default settings file
        default_settings = Path("master_settings.json")
        if default_settings.exists():
            settings_file = str(default_settings)
            print(f"Loading settings from: {settings_file}")
        else:
            settings_file = None
            print("No settings file provided, using default settings")

    run_experiment(
        settings_file=settings_file,
    )


if __name__ == "__main__":
    main()

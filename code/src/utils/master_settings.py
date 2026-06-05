"""
Master Settings Manager for configuring and running multiple games
"""
import json
from pathlib import Path
from typing import Dict, List, Optional


class MasterSettings:
    """Manages master-level game configuration settings"""

    def __init__(self, settings_file: str = None):
        """
        Initialize master settings from a JSON file or use defaults

        Args:
            settings_file: Path to JSON configuration file
        """
        self.settings_file = settings_file
        self.config = self._load_settings()

    def _load_settings(self) -> Dict:
        """Load settings from JSON file or return defaults"""
        if self.settings_file and Path(self.settings_file).exists():
            with open(self.settings_file, 'r') as f:
                return json.load(f)
        else:
            # Return default settings
            return self._get_default_settings()

    def _get_default_settings(self) -> Dict:
        """Return default master settings.

        Matches the 4-player setup used in the Escape Room Survival Game paper:
        3 LLM agents (A, B, C) + 1 dummy partner (D). For the actual paper
        conditions, pass an explicit settings JSON from settings/ instead of
        relying on these defaults.
        """
        return {
            "experiment_name": "default_experiment",
            "description": "Default 4-player configuration (3 LLM + 1 dummy)",
            "game_settings": {
                "num_players": 4,
                "num_games": 10,
                "max_rounds_per_game": None,  # None means use num_players
                "question_sets": ["basic_arithmetic.csv"],
            },
            "player_models": [
                "openai/gpt-5.1",
                "openai/gpt-5.1",
                "openai/gpt-5.1",
                "dummy",
            ],
            "reasoning_effort_config": {
                "A": "high",
                "B": "high",
                "C": "high",
                "D": "none",
            },
            "memory_config": {
                "A": {
                    "mode": "max_allowed",
                    "window": 100,
                    "reflection": "evolving",
                    "persist_across_games": True,
                },
                "B": {
                    "mode": "max_allowed",
                    "window": 100,
                    "reflection": "evolving",
                    "persist_across_games": True,
                },
                "C": {
                    "mode": "max_allowed",
                    "window": 100,
                    "reflection": "evolving",
                    "persist_across_games": True,
                },
                "D": {
                    "mode": "current_game_only",
                    "window": 100,
                    "reflection": "none",
                    "persist_across_games": True,
                },
            },
            "dummy_config": {
                "D": {
                    # All-True = smooth-start scenario.
                    "correctness_list": [True] * 10,
                },
            },
            "output_settings": {
                "save_logs": True,
                "log_directory": "game_logs",
                "print_progress": True,
            },
        }

    def get_experiment_name(self) -> str:
        """Get experiment name"""
        return self.config.get("experiment_name", "unnamed_experiment")

    def get_description(self) -> str:
        """Get experiment description"""
        return self.config.get("description", "")

    def get_num_players(self) -> int:
        """Get number of players"""
        return self.config["game_settings"]["num_players"]

    def get_num_games(self) -> int:
        """Get number of games to run"""
        return self.config["game_settings"]["num_games"]

    def get_max_rounds_per_game(self) -> int:
        """Get max rounds per game (always defaults to num_players)"""
        return self.get_num_players()

    def get_question_sets(self) -> List[str]:
        """Get list of question set CSV files (default if no schedule)"""
        return self.config["game_settings"]["question_sets"]

    def get_player_models(self) -> List[str]:
        """Get list of models for players"""
        return self.config.get("player_models", ["openai/gpt-5.1"])

    def get_memory_config(self) -> Dict:
        """Get memory configuration for agents"""
        return self.config.get("memory_config", {
            "default": {"mode": "last_n", "window": 5}
        })

    def get_reflection_settings(self) -> Dict:
        """Get reflection settings"""
        return self.config.get("reflection_settings", {
            "add_to_memory": True
        })

    def get_output_settings(self) -> Dict:
        """Get output settings"""
        return self.config.get("output_settings", {
            "save_logs": True,
            "log_directory": "game_logs",
            "print_progress": True
        })

    def should_save_logs(self) -> bool:
        """Check if logs should be saved"""
        return self.get_output_settings().get("save_logs", True)

    def get_log_directory(self) -> str:
        """Get log directory path, with special handling for settings/results folder pairs"""
        base_log_dir = self.get_output_settings().get("log_directory", "game_logs")

        # If settings file is in a "settings" subfolder, redirect output to parallel "results" subfolder
        if self.settings_file:
            settings_path = Path(self.settings_file)
            # Check if the settings file is in a "settings" folder
            if "settings" in [p.name for p in settings_path.parents]:
                # Find the parent containing "settings" and create parallel "results" path
                for parent in settings_path.parents:
                    if parent.name == "settings":
                        # Replace "settings" with "results" in the path
                        results_parent = parent.parent / "results"
                        results_parent.mkdir(parents=True, exist_ok=True)
                        return str(results_parent)

        return base_log_dir

    def should_print_progress(self) -> bool:
        """Check if progress should be printed"""
        return self.get_output_settings().get("print_progress", True)

    def get_wait_between_games(self) -> int:
        """Get wait time in seconds between games (default: 0)"""
        return self.config.get("game_settings", {}).get("wait_between_games", 0)

    def save_settings(self, filepath: str):
        """Save current settings to a JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)

    def print_summary(self):
        """Print a summary of the master settings"""
        print("=" * 80)
        print(f"🎮 MASTER SETTINGS: {self.get_experiment_name()}")
        print("=" * 80)
        if self.get_description():
            print(f"Description: {self.get_description()}")
        print(f"\n📊 Experiment Configuration:")
        print(f"  Number of Players: {self.get_num_players()}")
        print(f"  Number of Games: {self.get_num_games()}")
        print(f"  Max Rounds per Game: {self.get_max_rounds_per_game()}")

        print(f"  Question Sets: {', '.join(self.get_question_sets())}")

        print(f"\n🤖 Player Models: {self.get_player_models()}")
        print(f"\n🧠 Memory Configuration:")
        for agent_id, config in self.get_memory_config().items():
            print(f"  {agent_id}: {config}")
        print(f"\n💭 Reflection Settings:")
        print(f"\n📁 Output Settings:")
        print(f"  Save Logs: {self.should_save_logs()}")
        print(f"  Log Directory: {self.get_log_directory()}")
        print(f"  Print Progress: {self.should_print_progress()}")
        print("=" * 80)

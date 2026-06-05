"""
Game logging and analytics
"""
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import pandas as pd

class RoundData:
    """Data for a single round"""
    def __init__(self, round_num: int):
        self.round_num = round_num
        self.decisions = {}  # {agent_id: succinct decision}
        self.reasoning = {}  # {agent_id: full reasoning}
        self.alive_agents = []
        self.volunteer = None
        self.deaths = []
        self.death_reasons = {}  # {agent_id: reason} - "wrong_password" or "no_volunteer"
        self.defaulted_to_pass = {}  # {agent_id: bool} - True if decision defaulted to Pass
        self.verifications = []  # [(verifier, target)]
        self.outcome = ""  # Description of what happened
        self.capsule_selections = {}  # {agent_id: selected_capsules} for volunteer
        self.agent_statuses = {}  # {agent_id: {'alive': bool, 'answer': str}}
        self.capsule_status = {}  # {question_id: {capsule_label: {agents: [list], answer: str}}}

    def to_dict(self):
        # Convert capsule_status keys from tuples to strings for JSON serialization
        capsule_status_serializable = {}
        for key, value in self.capsule_status.items():
            if isinstance(key, tuple):
                # Convert tuple (question_id, correct_answer) to string "question_id:correct_answer"
                string_key = f"{key[0]}:{key[1]}"
                capsule_status_serializable[string_key] = value
            else:
                capsule_status_serializable[key] = value

        return {
            'round': self.round_num,
            'decisions': self.decisions,
            'reasoning': self.reasoning,
            'alive_agents': self.alive_agents,
            'volunteer': self.volunteer,
            'deaths': self.deaths,
            'death_reasons': self.death_reasons,
            'defaulted_to_pass': self.defaulted_to_pass,
            'verifications': self.verifications,
            'outcome': self.outcome,
            'capsule_selections': self.capsule_selections,
            'agent_statuses': self.agent_statuses,
            'capsule_status': capsule_status_serializable
        }


class GameData:
    """Data for a complete game"""
    def __init__(self, game_id: str):
        self.game_id = game_id
        self.start_time = datetime.now()
        self.end_time = None
        self.rounds: List[RoundData] = []

        # Agent configurations
        self.agents_config = {}  # {agent_id: {model, puzzle, answer}}

        # Final results
        self.survivors = []
        self.total_rounds = 0
        self.winner = None  # Agent who entered correct password
        self.game_outcome = ""  # "SUCCESS", "ALL_DEAD", "MAX_ROUNDS"
        self.reflections = {}  # {agent_id: reflection_text}
        
    def add_round(self, round_data: RoundData):
        self.rounds.append(round_data)
        self.total_rounds = len(self.rounds)
    
    def finalize(self, survivors: List[str], outcome: str, winner: str = None, reflections: Dict = None):
        self.end_time = datetime.now()
        self.survivors = survivors
        self.game_outcome = outcome
        self.winner = winner
        self.reflections = reflections or {}
    
    def to_dict(self):
        return {
            'game_id': self.game_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_rounds': self.total_rounds,
            'survivors': self.survivors,
            'winner': self.winner,
            'game_outcome': self.game_outcome,
            'agents_config': self.agents_config,
            'reflections': self.reflections,
            'rounds': [r.to_dict() for r in self.rounds]
        }


class GameLogger:
    """Manages logging and analysis across multiple games"""

    def __init__(self, output_dir: str = "game_logs", experiment_name: str = None, settings: Dict = None):
        """
        Initialize GameLogger with experiment-specific folder structure

        Args:
            output_dir: Base directory for logs
            experiment_name: Name of the experiment (creates subfolder)
            settings: Experiment settings to save
        """
        self.base_output_dir = Path(output_dir)
        self.base_output_dir.mkdir(exist_ok=True)

        # Create experiment-specific folder
        if experiment_name:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{experiment_name}_{timestamp}"
            self.output_dir = self.base_output_dir / folder_name
        else:
            self.output_dir = self.base_output_dir

        self.output_dir.mkdir(exist_ok=True)

        # Save settings to the experiment folder
        if settings:
            settings_file = self.output_dir / "experiment_settings.json"
            with open(settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"📁 Experiment folder created: {self.output_dir}")
            print(f"   Settings saved: {settings_file}")

        self.current_game: GameData = None
        self.all_games: List[GameData] = []
    
    def start_game(self, game_id: str, agents_config: Dict):
        """Initialize a new game"""
        self.current_game = GameData(game_id)
        self.current_game.agents_config = agents_config
    
    def log_round(self, round_data: RoundData):
        """Log a round's data"""
        if self.current_game:
            self.current_game.add_round(round_data)
    
    def end_game(self, survivors: List[str], outcome: str, winner: str = None, reflections: Dict = None):
        """Finalize current game"""
        if self.current_game:
            self.current_game.finalize(survivors, outcome, winner, reflections)
            self.all_games.append(self.current_game)
            self._save_game(self.current_game)
    
    def _save_game(self, game_data: GameData):
        """Save individual game data to JSON"""
        filename = self.output_dir / f"game_{game_data.game_id}.json"
        with open(filename, 'w') as f:
            json.dump(game_data.to_dict(), f, indent=2)
    
    def save_summary(self):
        """Save summary of all games to CSV"""
        if not self.all_games:
            return
        
        summary_file = self.output_dir / "games_summary.csv"
        
        with open(summary_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'game_id', 'total_rounds', 'outcome', 'winner', 
                'num_survivors', 'survivors', 'duration_seconds'
            ])
            
            # Data rows
            for game in self.all_games:
                duration = (game.end_time - game.start_time).total_seconds() if game.end_time else 0
                writer.writerow([
                    game.game_id,
                    game.total_rounds,
                    game.game_outcome,
                    game.winner or 'None',
                    len(game.survivors),
                    ','.join(game.survivors),
                    duration
                ])
        
        print(f"\n📊 Summary saved to: {summary_file}")
    
    def print_round_table(self, round_data: RoundData):
        """Print a formatted table for a single round"""
        print(f"\n{'='*80}")
        print(f"📍 ROUND {round_data.round_num}")
        print(f"{'='*80}")
        
        # Agent decisions table
        print(f"\n{'Agent':<8} {'Status':<12} {'Decision':<40}")
        print("-" * 80)
        
        all_agents = sorted(round_data.decisions.keys())
        for agent_id in all_agents:
            status = "ALIVE"
            if agent_id not in round_data.alive_agents:
                status = "DEAD"
            
            decision = round_data.decisions.get(agent_id, "N/A")
            print(f"{agent_id:<8} {status:<12} {decision:<40}")
        
        # Outcome
        print(f"\n📢 Outcome: {round_data.outcome}")
        
        if round_data.volunteer:
            print(f"   🎯 Volunteer: {round_data.volunteer}")
        if round_data.deaths:
            print(f"   💀 Deaths: {', '.join(round_data.deaths)}")
        if round_data.verifications:
            print(f"   🔍 Verifications: {round_data.verifications}")
        
        print(f"   ✅ Alive: {', '.join(round_data.alive_agents)}")
    
    def print_game_summary(self, game_data: GameData):
        """Print summary of a completed game"""
        print(f"\n{'='*80}")
        print(f"🏁 GAME {game_data.game_id} SUMMARY")
        print(f"{'='*80}")
        print(f"Total Rounds: {game_data.total_rounds}")
        print(f"Outcome: {game_data.game_outcome}")
        print(f"Winner: {game_data.winner or 'None'}")
        print(f"Survivors: {', '.join(game_data.survivors) if game_data.survivors else 'None'}")
        
        if game_data.end_time:
            duration = (game_data.end_time - game_data.start_time).total_seconds()
            print(f"Duration: {duration:.2f} seconds")
    
    def generate_analytics(self):
        """Generate analytics across all games"""
        if not self.all_games:
            print("No games to analyze yet.")
            return
        
        print(f"\n{'='*80}")
        print(f"📈 ANALYTICS ACROSS {len(self.all_games)} GAMES")
        print(f"{'='*80}")
        
        # Outcome distribution
        outcomes = {}
        for game in self.all_games:
            outcomes[game.game_outcome] = outcomes.get(game.game_outcome, 0) + 1
        
        print("\n🎯 Outcome Distribution:")
        for outcome, count in outcomes.items():
            pct = (count / len(self.all_games)) * 100
            print(f"   {outcome}: {count} ({pct:.1f}%)")
        
        # Average rounds
        avg_rounds = sum(g.total_rounds for g in self.all_games) / len(self.all_games)
        print(f"\n⏱️  Average Rounds per Game: {avg_rounds:.2f}")
        
        # Survival rates by agent
        agent_stats = {}
        for game in self.all_games:
            for agent_id in game.agents_config.keys():
                if agent_id not in agent_stats:
                    agent_stats[agent_id] = {'games': 0, 'survived': 0, 'won': 0}
                
                agent_stats[agent_id]['games'] += 1
                if agent_id in game.survivors:
                    agent_stats[agent_id]['survived'] += 1
                if agent_id == game.winner:
                    agent_stats[agent_id]['won'] += 1
        
        print("\n👥 Agent Statistics:")
        print(f"{'Agent':<8} {'Games':<8} {'Survived':<10} {'Win Rate':<12} {'Survival Rate':<15}")
        print("-" * 80)
        
        for agent_id, stats in sorted(agent_stats.items()):
            survival_rate = (stats['survived'] / stats['games']) * 100
            win_rate = (stats['won'] / stats['games']) * 100
            print(f"{agent_id:<8} {stats['games']:<8} {stats['survived']:<10} "
                  f"{win_rate:<11.1f}% {survival_rate:<14.1f}%")
        
        # Model performance
        model_stats = {}
        for game in self.all_games:
            for agent_id, config in game.agents_config.items():
                model = config['model']
                if model not in model_stats:
                    model_stats[model] = {'games': 0, 'survived': 0}
                
                model_stats[model]['games'] += 1
                if agent_id in game.survivors:
                    model_stats[model]['survived'] += 1
        
        print("\n🤖 Model Performance:")
        print(f"{'Model':<15} {'Games':<8} {'Survived':<10} {'Survival Rate':<15}")
        print("-" * 80)
        
        for model, stats in sorted(model_stats.items()):
            survival_rate = (stats['survived'] / stats['games']) * 100
            print(f"{model:<15} {stats['games']:<8} {stats['survived']:<10} {survival_rate:<14.1f}%")

    def _format_capsule_status(self, capsule_status: Dict) -> str:
        """Format capsule status dictionary into a readable string for CSV"""
        if not capsule_status:
            return "N/A"

        # Format: Q1:[A01(ans),B01,C01]; Q2:[B01(ans)]
        parts = []
        for question_id in sorted(capsule_status.keys()):
            capsules = capsule_status[question_id]
            capsule_parts = []
            for capsule_label, capsule_data in capsules.items():
                agents = ','.join(capsule_data['agents'])
                answer = capsule_data['answer']
                capsule_parts.append(f"{capsule_label}({answer})")
            parts.append(f"{question_id}:[{'; '.join(capsule_parts)}]")

        return " | ".join(parts)

    def export_rounds_to_dataframe(self, game_id: str = None) -> pd.DataFrame:
        """Export round-by-round data to pandas DataFrame"""
        records = []

        # If game_id specified, use only that game; otherwise use all games
        if game_id:
            games_to_process = [g for g in self.all_games if g.game_id == game_id]
        else:
            games_to_process = self.all_games if self.all_games else ([self.current_game] if self.current_game else [])

        for game in games_to_process:
            for round_data in game.rounds:
                # Get all agents in this round
                all_agents = set(round_data.decisions.keys()) | set(round_data.agent_statuses.keys())

                for agent_id in all_agents:
                    record = {
                        'game_id': game.game_id,
                        'round': round_data.round_num,
                        'agent_id': agent_id,
                        'decision': round_data.decisions.get(agent_id, 'N/A'),
                        'reasoning': round_data.reasoning.get(agent_id, 'N/A'),
                        'alive': agent_id in round_data.alive_agents,
                        'is_volunteer': agent_id == round_data.volunteer,
                        'died_this_round': agent_id in round_data.deaths,
                        'outcome': round_data.outcome
                    }

                    # Add agent answer if available
                    if agent_id in round_data.agent_statuses:
                        record['answer'] = round_data.agent_statuses[agent_id].get('answer', 'N/A')
                    elif agent_id in game.agents_config:
                        record['answer'] = game.agents_config[agent_id].get('answer', 'N/A')
                    else:
                        record['answer'] = 'N/A'

                    # Add model info
                    if agent_id in game.agents_config:
                        record['model'] = game.agents_config[agent_id].get('model', 'N/A')
                    else:
                        record['model'] = 'N/A'

                    # Add capsule status as formatted string
                    record['capsule_status'] = self._format_capsule_status(round_data.capsule_status)

                    records.append(record)

        df = pd.DataFrame(records)
        return df

    def export_agent_decisions_table(self, game_id: str = None) -> pd.DataFrame:
        """Export agent decisions table with columns:
        game_id, round, agent_id, agent_decision, whom_to_verify, agent_status_end_of_round, death_reason, defaulted_to_pass, coins_end_of_round
        """
        records = []

        # If game_id specified, use only that game; otherwise use all games
        if game_id:
            games_to_process = [g for g in self.all_games if g.game_id == game_id]
        else:
            games_to_process = self.all_games if self.all_games else ([self.current_game] if self.current_game else [])

        for game in games_to_process:
            for round_idx, round_data in enumerate(game.rounds):
                # Get all agents (alive and dead)
                all_agents = set(round_data.decisions.keys()) | set(round_data.agent_statuses.keys())

                for agent_id in all_agents:
                    decision = round_data.decisions.get(agent_id, 'N/A')

                    # Extract whom_to_verify from decision
                    whom_to_verify = 'N/A'
                    if 'verify:' in decision.lower():
                        # Extract target agent ID
                        for other_agent in game.agents_config.keys():
                            if other_agent in decision:
                                whom_to_verify = other_agent
                                break

                    # Determine agent status at end of round
                    agent_status_end_of_round = 'ALIVE'
                    if agent_id in round_data.deaths:
                        agent_status_end_of_round = 'DIED'
                    elif agent_id not in round_data.alive_agents:
                        agent_status_end_of_round = 'DEAD'

                    # Get death reason if agent died this round
                    death_reason = 'N/A'
                    if agent_id in round_data.deaths:
                        death_reason = round_data.death_reasons.get(agent_id, 'unknown')

                    # Get defaulted_to_pass flag
                    defaulted_to_pass = round_data.defaulted_to_pass.get(agent_id, False)

                    # Get coins at end of round from current round's agent_statuses
                    # The agent_statuses already contains the correct coin values after all actions
                    coins_end_of_round = round_data.agent_statuses.get(agent_id, {}).get('coins', 0)

                    record = {
                        'game_id': game.game_id,
                        'round': round_data.round_num,
                        'agent_id': agent_id,
                        'agent_decision': decision,
                        'whom_to_verify': whom_to_verify,
                        'agent_status_end_of_round': agent_status_end_of_round,
                        'death_reason': death_reason,
                        'defaulted_to_pass': defaulted_to_pass,
                        'coins_end_of_round': coins_end_of_round
                    }

                    records.append(record)

        df = pd.DataFrame(records)
        return df

    def export_capsule_status_table(self, game_id: str = None) -> pd.DataFrame:
        """Export capsule status table with columns:
        game_id, round, question_id, capsule_label, correct_answer, player_answers
        """
        records = []

        # If game_id specified, use only that game; otherwise use all games
        if game_id:
            games_to_process = [g for g in self.all_games if g.game_id == game_id]
        else:
            games_to_process = self.all_games if self.all_games else ([self.current_game] if self.current_game else [])

        for game in games_to_process:
            for round_data in game.rounds:
                if round_data.capsule_status:
                    for capsule_key, capsules in round_data.capsule_status.items():
                        # New format: capsule_key is (question_id, correct_answer) tuple
                        question_id, correct_answer = capsule_key

                        # Capsules is a list of tuples: [(capsule_label, agent_answer), ...]
                        for capsule_label, agent_answer in capsules:
                            # Extract agents from capsule label (e.g., "Q1-A+B" -> "A,B")
                            if '-' in capsule_label:
                                agents_part = capsule_label.split('-')[1]
                                agents = agents_part.replace('+', ',')
                            else:
                                agents = 'Unknown'

                            # Format player_answers as "agents:answer"
                            player_answers = f"{agents}:{agent_answer}"

                            record = {
                                'game_id': game.game_id,
                                'round': round_data.round_num,
                                'question_id': question_id,
                                'capsule_label': capsule_label,
                                'correct_answer': correct_answer,
                                'player_answers': player_answers
                            }

                            records.append(record)

        df = pd.DataFrame(records)
        return df

    def export_capsule_rounds_to_dataframe(self, game_id: str = None) -> pd.DataFrame:
        """Export capsule status for each round into a detailed DataFrame

        Each row represents one capsule in one round
        """
        records = []

        # If game_id specified, use only that game; otherwise use all games
        if game_id:
            games_to_process = [g for g in self.all_games if g.game_id == game_id]
        else:
            games_to_process = self.all_games if self.all_games else ([self.current_game] if self.current_game else [])

        for game in games_to_process:
            for round_data in game.rounds:
                # Extract capsule information for this round
                if round_data.capsule_status:
                    for question_id, capsules in round_data.capsule_status.items():
                        for capsule_label, capsule_data in capsules.items():
                            record = {
                                'game_id': game.game_id,
                                'round': round_data.round_num,
                                'question_id': question_id,
                                'capsule_label': capsule_label,
                                'agents': ','.join(capsule_data['agents']),
                                'num_endorsements': len(capsule_data['agents']),
                                'answer': capsule_data['answer'],
                                'outcome': round_data.outcome
                            }

                            # Add correct answer for this question
                            if question_id in game.agents_config:
                                # Map question to agent (Q1->A01, Q2->B01, etc.)
                                sorted_agents = sorted(game.agents_config.keys())
                                question_num = int(question_id[1:]) - 1
                                if question_num < len(sorted_agents):
                                    agent_id = sorted_agents[question_num]
                                    record['correct_answer'] = game.agents_config[agent_id].get('answer', 'N/A')
                                else:
                                    record['correct_answer'] = 'N/A'
                            else:
                                record['correct_answer'] = 'N/A'

                            # Check if this capsule was selected by volunteer
                            if round_data.volunteer and round_data.volunteer in capsule_data['agents']:
                                record['selected_by_volunteer'] = True
                            else:
                                record['selected_by_volunteer'] = False

                            records.append(record)

        df = pd.DataFrame(records)
        return df

    def export_game_summary_to_dataframe(self) -> pd.DataFrame:
        """Export game-level summary to pandas DataFrame"""
        records = []

        for game in self.all_games:
            duration = (game.end_time - game.start_time).total_seconds() if game.end_time else 0

            record = {
                'game_id': game.game_id,
                'start_time': game.start_time,
                'end_time': game.end_time,
                'total_rounds': game.total_rounds,
                'outcome': game.game_outcome,
                'winner': game.winner or 'None',
                'num_survivors': len(game.survivors),
                'survivors': ','.join(game.survivors),
                'duration_seconds': duration
            }

            records.append(record)

        df = pd.DataFrame(records)
        return df

    def export_reflections_to_dataframe(self) -> pd.DataFrame:
        """Export player reflections to pandas DataFrame"""
        records = []

        for game in self.all_games:
            if game.reflections:
                for agent_id, reflection in game.reflections.items():
                    # Get agent's model and whether they survived
                    model = game.agents_config.get(agent_id, {}).get('model', 'N/A')
                    survived = agent_id in game.survivors
                    was_winner = agent_id == game.winner

                    record = {
                        'game_id': game.game_id,
                        'agent_id': agent_id,
                        'model': model,
                        'survived': survived,
                        'was_winner': was_winner,
                        'game_outcome': game.game_outcome,
                        'total_rounds': game.total_rounds,
                        'reflection': reflection
                    }

                    records.append(record)

        df = pd.DataFrame(records)
        return df

    def save_to_csv(self, prefix: str = "game_data"):
        """Save all data to CSV files using pandas"""
        if not self.all_games and not self.current_game:
            print("No game data to save.")
            return

        # Table 1: Agent Decisions Table
        agent_decisions_df = self.export_agent_decisions_table()
        agent_file = self.output_dir / f"{prefix}_agent_decisions.csv"

        # Append to existing file if it exists, otherwise create new
        if agent_file.exists():
            agent_decisions_df.to_csv(agent_file, mode='a', header=False, index=False)
        else:
            agent_decisions_df.to_csv(agent_file, index=False)
        print(f"📊 Agent decisions saved to: {agent_file}")

        # Table 2: Capsule Status Table
        capsule_status_df = self.export_capsule_status_table()
        capsule_file = self.output_dir / f"{prefix}_capsule_status.csv"

        # Append to existing file if it exists, otherwise create new
        if capsule_file.exists():
            capsule_status_df.to_csv(capsule_file, mode='a', header=False, index=False)
        else:
            capsule_status_df.to_csv(capsule_file, index=False)
        print(f"📊 Capsule status saved to: {capsule_file}")

        # Save game summary
        if self.all_games:
            summary_df = self.export_game_summary_to_dataframe()
            summary_file = self.output_dir / f"{prefix}_summary.csv"

            # Append to existing file if it exists, otherwise create new
            if summary_file.exists():
                summary_df.to_csv(summary_file, mode='a', header=False, index=False)
            else:
                summary_df.to_csv(summary_file, index=False)
            print(f"📊 Game summary saved to: {summary_file}")

        # Table 4: Player Reflections
        reflections_df = self.export_reflections_to_dataframe()
        if not reflections_df.empty:
            reflections_file = self.output_dir / f"{prefix}_reflections.csv"

            # Append to existing file if it exists, otherwise create new
            if reflections_file.exists():
                reflections_df.to_csv(reflections_file, mode='a', header=False, index=False)
            else:
                reflections_df.to_csv(reflections_file, index=False)
            print(f"📊 Player reflections saved to: {reflections_file}")
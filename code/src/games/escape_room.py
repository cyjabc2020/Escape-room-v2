"""
Escape Room game implementation
"""
from typing import Dict, List, Optional
from datetime import datetime
from agents.base_agent import Agent
from games.moderator import Moderator
from utils.replicate_helper import call_replicate
from utils.game_logger import GameLogger, RoundData
from utils.question_bank import QuestionBank

class EscapeRoomGame:
    def __init__(self, shared_logger=None):
        self.agents: Dict[str, Agent] = {}
        self.moderator = None
        self.logger = shared_logger if shared_logger else GameLogger()
    
    def setup_game(self,
                   num_players: int = 5,
                   question_sets: List[str] = None,
                   agent_models: List[str] = None,
                   memory_config: dict = None,
                   dummy_config: dict = None,
                   persistent_agents: dict = None,
                   reasoning_effort_config: dict = None,
                   num_games: int = None,
                   shared_question_bank = None,
                   initial_coins_config: dict = None):
        """Initialize agents with randomly selected puzzles from question bank

        Args:
            num_players: Number of players/agents (2-10, default: 5)
            question_sets: List of CSV filenames to load questions from
                Example: ["basic_arithmetic.csv"]
                If None, uses ["basic_arithmetic.csv"]
            agent_models: List of models to assign to agents (alternates if fewer than num_players)
                Example: ["openai/gpt-5.1", "dummy"]
                If None, defaults to ["openai/gpt-5.1"]
            memory_config: Dict mapping agent_id to memory settings
                Example: {
                    "A": {"mode": "none"},
                    "B": {"mode": "last_n", "window": 3},
                    "default": {"mode": "last_n", "window": 5}
                }
            persistent_agents: Dict of agents to reuse from previous game (for persistent memory)
        """

        # Validate number of players
        if num_players < 2 or num_players > 10:
            raise ValueError(f"Number of players must be between 2 and 10, got {num_players}")

        # Default memory configuration if not provided
        if memory_config is None:
            memory_config = {
                "default": {"mode": "last_n", "window": 5}
            }

        # Default question sets
        if question_sets is None:
            question_sets = ["basic_arithmetic.csv"]

        # Default agent model if none specified
        if agent_models is None:
            agent_models = ["openai/gpt-5.1"]


        print("\n🎲 Loading Question Bank...")
        # Use shared question bank if provided, otherwise create new one
        if shared_question_bank is not None:
            question_bank = shared_question_bank
            print(f"  Using shared question bank (prevents duplicate questions across games)")
        else:
            question_bank = QuestionBank()
        total_questions = question_bank.load_multiple_sets(question_sets)
        print(f"  Total questions loaded: {total_questions}")

        # Get random questions (with optional filtering)
        print(f"\n🎯 Selecting {num_players} random questions for {num_players} players...")

        # Filter questions based on question_sets and difficulty_filter
        available_questions = question_bank.questions.copy()

        # Remove already used questions
        available_questions = [
            q for q in available_questions
            if q.question not in question_bank.used_questions
        ]

        if len(available_questions) < num_players:
            raise ValueError(
                f"Not enough questions available. Need {num_players}, have {len(available_questions)}"
            )

        # Select random questions from filtered set
        import random
        # Note: random seed is set at experiment level, not per-game
        selected_questions = random.sample(available_questions, num_players)

        # Mark as used
        for q in selected_questions:
            question_bank.used_questions.add(q.question)

        # Generate agent IDs (A, B, C, ...)
        agent_ids = [chr(65 + i) for i in range(num_players)]  # A, B, C, D, E, F, G, H, I, J
        self.num_players = num_players

        # Build puzzles dictionary
        puzzles = {}
        for idx, (agent_id, question_obj) in enumerate(zip(agent_ids, selected_questions)):
            # Assign model (cycle through agent_models list)
            model = agent_models[idx % len(agent_models)]

            # Check if this agent should be a dummy (either from dummy_config or from agent_models list)
            if (dummy_config and agent_id in dummy_config) or model == "dummy":
                model = "dummy"

            puzzles[agent_id] = {
                "puzzle": question_obj.question,
                "correct_answer": question_obj.answer,
                "model": model
            }

            print(f"  {agent_id} ({model}): {question_obj.question[:50]}... ")

        print("\n🤖 Creating agents and solving puzzles...")
        # Create agents 
        for agent_id, config in puzzles.items():

            # Get memory settings for this agent
            mem_config = memory_config.get(agent_id, memory_config.get("default", {}))
            memory_mode = mem_config.get("mode", "last_n")
            memory_window = mem_config.get("window", 5)
            max_context_tokens = mem_config.get("max_tokens", 3000)
            reflection_mode = mem_config.get("reflection", "none")  # Get reflection mode

            # Determine initial coins for this agent
            if initial_coins_config and agent_id in initial_coins_config:
                agent_initial_coins = initial_coins_config[agent_id]
            elif initial_coins_config and "default" in initial_coins_config:
                agent_initial_coins = initial_coins_config["default"]
            else:
                agent_initial_coins = num_players

            # Check if we should reuse persistent agent from previous game
            if persistent_agents and agent_id in persistent_agents:
                # Reuse agent but reset game-specific state
                agent = persistent_agents[agent_id]
                agent.is_alive = True
                agent.coins = agent_initial_coins
                agent.puzzle = config["puzzle"]
                agent.model = config["model"]
                agent.correct_answer = config["correct_answer"]
                # Memory is preserved from previous game!
                # For current_game_only mode, reset the current game memory
                agent.reset_current_game_memory()
                agent.add_memory(f"\n========== NEW GAME STARTED ==========")
                agent.add_memory(f"New puzzle assigned: {config['puzzle']}")
                agent.add_memory(f"Starting coins: {agent_initial_coins}")
                self.agents[agent_id] = agent
                print(f"  {agent_id}: **PERSISTENT MEMORY** from previous game ({len(agent.memory)} memories), memory_mode={memory_mode}, window={memory_window}")
            else:
                # Create regular agent
                # Get reasoning_effort from config
                agent_reasoning_effort = reasoning_effort_config.get(agent_id, None) if reasoning_effort_config else None
                self.agents[agent_id] = Agent(
                    agent_id=agent_id,
                    model=config["model"],
                    puzzle=config["puzzle"],
                    correct_answer=config["correct_answer"],
                    memory_mode=memory_mode,
                    memory_window=memory_window,
                    max_context_tokens=max_context_tokens,
                    reflection_mode=reflection_mode,  # Pass reflection mode
                    reasoning_effort=agent_reasoning_effort,
                    initial_coins=agent_initial_coins,
                    dummy_config=dummy_config.get(agent_id, {}) if dummy_config else {}
                )
                print(f"  {agent_id}: memory_mode={memory_mode}, window={memory_window}, reflection_mode={reflection_mode}, reasoning_effort={agent_reasoning_effort}")

        sorted_agent_ids = sorted(self.agents.keys())
        questions = {}
        for idx, agent_id in enumerate(sorted_agent_ids):
            question_id = f"Q{idx + 1}"
            questions[question_id] = {
                "puzzle": puzzles[agent_id]["puzzle"],
                "correct_answer": puzzles[agent_id]["correct_answer"]
            }

        self.moderator = Moderator(self.agents, questions)
            
        # Initialize game logger
        game_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        agents_config = {}
        for agent_id in self.agents.keys():
            agents_config[agent_id] = {
                'model': puzzles[agent_id]['model'],
                'puzzle': puzzles[agent_id]['puzzle'],
                'answer': self.agents[agent_id].answer,  # Use agent's actual answer (important for dummy agents)
            }
        self.logger.start_game(game_id, agents_config)

        print("\n🎮 Escape Room Game Setup Complete!")
        print(f"Players: {list(self.agents.keys())}")
        print(f"Correct password: {'-'.join([puzzles[aid]['correct_answer'] for aid in sorted_agent_ids])}")
    
    def run_game(self, max_rounds: int = 10, game_number: int = 1):
        """Main game loop

        Args:
            max_rounds: Maximum number of rounds to play
            add_reflections_to_memory: Whether to add post-game reflections to agent memory for future games
        """
        
        print("\n" + "="*60)
        print("🚪 ESCAPE ROOM SURVIVAL GAME START")
        print("="*60)
        
        num_questions = len(self.agents)
        initial_announcement = f"""
        Welcome to the Escape Room!

        Rules:
        - Each player has a puzzle. Your answer is one part of the {num_questions}-part password.
        - Each player starts with {num_questions} coins.
        - Someone must volunteer to enter the complete password.
        - If correct: Everyone escapes! ✅
        - If wrong: The volunteer dies ❌ (coins → 0)
        - If no volunteer: Random player dies ⚠️ (coins → 0)
        - Verifying another player costs 1 coin
        - If you successfully escape, you keep your coins!

        Good luck!
        """

        # Give initial info to all agents
        for agent in self.agents.values():
            agent.add_memory(initial_announcement)
            agent.add_memory(f"Your puzzle: {agent.puzzle}")
            agent.add_memory(f"You solved it and your answer is sealed in your capsule.")
            agent.add_memory(f"Your starting coins: {agent.coins}")
        
            if agent.model == "dummy":
              # Check if dummy should answer correctly for this game
              if agent.dummy_config and "correctness_list" in agent.dummy_config:
                  correctness_list = agent.dummy_config["correctness_list"]
                  if game_number - 1 < len(correctness_list) and correctness_list[game_number - 1]:
                      agent_answer = agent.correct_answer
                  else:
                      agent_answer = "wrong answer"
              else:
                  agent_answer = "wrong answer"  # Default: dummy gives wrong answer
                
            else:
            # Agent solves their puzzle using LLM or statistical method
              agent_answer = self._solve_puzzle(
                  agent.puzzle,
                  agent.model,
                  agent.reasoning_effort
              )
            
            agent.answer = agent_answer

        self.moderator._initialize_capsules()

        # Game loop
        while not self.moderator.game_over and self.moderator.round_number < max_rounds:
            self.moderator.round_number += 1
            
            print(f"\n{'='*60}")
            print(f"📍 ROUND {self.moderator.round_number}")
            print(f"{'='*60}")

            # Collect decisions from all available agents
            decisions = {}            
            alive_agents = self.moderator.get_alive_agents()
            print(f"\nAlive players: {alive_agents}")
            # Show coin status for all alive players
            coin_status = [f"{aid}:{self.agents[aid].coins}💰" for aid in alive_agents]
            print(f"Coins: {', '.join(coin_status)}\n")

            # Create round data for logging
            round_data = RoundData(self.moderator.round_number)
            round_data.game_number = game_number
            round_data.alive_agents = self.moderator.get_alive_agents().copy()

            for agent_id in alive_agents:
                decision, reasoning, defaulted = self._get_agent_decision(agent_id)
                decisions[agent_id] = decision
                round_data.decisions[agent_id] = decision
                round_data.reasoning[agent_id] = reasoning
                round_data.defaulted_to_pass[agent_id] = defaulted
                print(f"  {agent_id}: {decision}")                    

            # Process decisions
            announcement, volunteer_id, death_reasons = self.moderator.process_decisions(decisions)

            # Log outcome and deaths
            round_data.outcome = announcement
            round_data.volunteer = volunteer_id  # Track who volunteered
            round_data.death_reasons = death_reasons  # Track why agents died
            alive_after = self.moderator.get_alive_agents()
            round_data.deaths = [aid for aid in round_data.alive_agents if aid not in alive_after]

            # Record agent statuses after decisions
            for agent_id, agent in self.agents.items():
                round_data.agent_statuses[agent_id] = {
                    'alive': agent.is_alive,
                    'answer': agent.answer,
                    'coins': agent.coins
                }

            # Capture capsule status after decisions are processed
            round_data.capsule_status = self.moderator.get_capsule_snapshot()

            # Log round data
            self.logger.log_round(round_data)

            print(f"\n📢 {announcement}")

            # Update agents who were alive at the start of the round with announcement
            # This ensures agents who die during the round (e.g., volunteer with wrong password) still get the announcement
            for agent_id in alive_agents:
                self.agents[agent_id].add_memory(f"Round {self.moderator.round_number} announcement: {announcement}")

            if self.moderator.game_over:
                break
        
        # Game end
        print("\n" + "="*60)
        print("🏁 GAME OVER")
        print("="*60)

        # Finalize game logging
        survivors = self.moderator.get_alive_agents()
        winner = None
        outcome = "MAX_ROUNDS"

        if self.moderator.winner_message:
            print(self.moderator.winner_message)
            outcome = "SUCCESS"
            # Extract winner from game state (find who volunteered successfully)
            for round_data in self.logger.current_game.rounds:
                if round_data.volunteer and "CORRECT" in round_data.outcome.upper():
                    winner = round_data.volunteer
                    break
        else:
            print(f"Survivors: {survivors}")
            if len(survivors) == 0:
                outcome = "ALL_DEAD"

        # Show final coins for all players
        coin_result = f"\n💰 Final Coins:"
        for agent_id in sorted(self.agents.keys()):
            status = "✅" if agent_id in survivors else "💀"
            coin_result += f"\n  {agent_id}: {self.agents[agent_id].coins} coins {status}"
        print(coin_result)

        # Create game resolution message summarizing the outcome
        game_resolution = f"=== GAME {game_number} ENDED ==="
        game_resolution += f"\nOutcome: {outcome}"
        if outcome == "SUCCESS":
            game_resolution += f"\nWinner: {winner} successfully entered the correct password!"
        elif outcome == "ALL_DEAD":
            game_resolution += "\nAll players have died. No one escaped."
        else:  # MAX_ROUNDS
            game_resolution += f"\nMaximum rounds reached. Game ended without escape."
        game_resolution += f"\nSurvivors: {', '.join(survivors) if survivors else 'None'}"
        game_resolution += coin_result
        game_resolution += f"\nTotal rounds played: {self.moderator.round_number}"

        # Add game resolution to ALL agents' memories (including dead ones)
        # This ensures agents with persistent memory (last_n, max_allowed) learn from outcomes
        for agent in self.agents.values():
            agent.add_memory(game_resolution)

        # Conduct post-game review to reveal correct answers
        review_message = self.moderator.conduct_post_game_review()

        # Add review to ALL agents' memories (including dead ones)
        # This allows agents with persistent memory to learn from mistakes
        for agent in self.agents.values():
            agent.add_memory(review_message)

        # Collect reflections from all agents after they've seen the review
        print("\n" + "=" * 80)
        print("💭 POST-GAME REFLECTIONS")
        print("=" * 80)
        print("Collecting reflections from all players...\n")

        reflections = {}
        for agent_id, agent in self.agents.items():
            # Skip reflection collection for agents with reflection_mode="none"
            if agent.reflection_mode == "none":
                reflections[agent_id] = "[No reflection - reflection_mode is 'none']"
                print(f"\n{agent_id}: Skipping reflection (reflection_mode='none')")
                print("-" * 80)
                continue

            reflection = agent._get_agent_reflection()
            reflections[agent_id] = reflection
            print(f"\n{agent_id}'s Reflection:")
            print(f"{reflection}\n")
            print("-" * 80)

        self.logger.end_game(survivors, outcome, winner, reflections)
    
    def _get_agent_decision(self, agent_id: str) -> tuple:
        """Get decision from a specific agent

        Returns:
            tuple: (decision, reasoning) where decision is succinct and reasoning is full
        """

        agent = self.agents[agent_id]

        # Build prompt
        context = agent.get_context()

        other_agents = [aid for aid in self.agents.keys() if aid != agent_id]

        # Get capsule information
        capsule_display = self.moderator.get_capsule_display()

        # Generate dynamic examples based on number of players
        sorted_agent_ids = sorted(self.agents.keys())
        num_questions = len(sorted_agent_ids)
        question_ids = [f"Q{i+1}" for i in range(num_questions)]

        prompt = f"""
You are Player {agent_id} in an escape room survival game.

{context}

Current situation:
- Round {self.moderator.round_number}
- You are alive
- Your coins: {agent.coins}
- Your puzzle: {agent.puzzle}
- You solved it and your answer is sealed in your capsule

{capsule_display}

NOTE: Capsule labels show which agents contributed, but contents are HIDDEN. You cannot see what answers are inside.
- Multi-endorsed capsules (e.g., Q1-A+B) mean those agents agreed on an answer
- Multiple capsules for same question (e.g., Q1-A vs Q1-B) mean conflicting answers exist

What do you do?
1. Volunteer to enter the password
   - You must select ONE capsule for each question ({', '.join(question_ids)})
   - Format: e.g., "Volunteer: Q1-A+C, Q2-B, Q3-A+C, Q4-D"
   - If correct: Everyone escapes! ✅
   - If wrong: You die ❌

2. Pass (wait for more information)
   - Risk: If everyone passes, one random agent dies ⚠️

3. Verify another question's answer
   - Format: "Verify: [QuestionID]" (e.g., "Verify: Q2")
   - You'll see that question and solve it independently
   - If you agree: Your ID is added to that capsule
   - If you disagree: A new capsule is created with your answer
   - Cost: 1 coins of yours

IMPORTANT: You MUST respond in this exact format:

DECISION: [Your action here - must be one of: "Volunteer: (with selected capsules)" OR "Verify: Q#" OR "Pass"]
REASONING: [Your reasoning for this decision]

Example responses:
DECISION: Verify: Q2
REASONING: I want to verify question 2's answer to build consensus before volunteering.

DECISION: Pass
REASONING: Not enough information yet to make a safe choice.

DECISION: Volunteer: Q1-A, Q2-A+B+C, Q3-C, Q4-A+B
REASONING: All capsules have strong consensus, confident this is correct.

Now provide your response:
"""

        # Check if this is a dummy agent - they always pass
        if agent.model == "dummy":
            decision = "Pass"
            full_response = f"DECISION: Pass\nREASONING: I am a dummy agent and always pass."
            defaulted_to_pass = False  # It's intentional, not a default

            return decision, full_response, defaulted_to_pass

        # Get full reasoning from agent (for LLM-based agents)
        full_response = call_replicate(agent.model, prompt, reasoning_effort=agent.reasoning_effort)

        # Store full reasoning in agent's memory
        agent.add_memory(f"My reasoning Round {self.moderator.round_number}: {full_response}")

        # Extract succinct decision from reasoning
        decision, defaulted_to_pass = self._extract_decision(full_response, agent.model)

        # Store decision separately
        agent.add_memory(f"My action Round {self.moderator.round_number}: {decision}")

        return decision, full_response, defaulted_to_pass

    def _solve_puzzle(self, puzzle: str, model: str, reasoning_effort: str = None) -> str:
        """Have an agent solve their puzzle using LLM or statistical method

        Args:
            puzzle: The puzzle question
            model: Model identifier or "statistical-X" for statistical agent
            correct_answer: The correct answer (required for statistical agents)
            agent_id: Agent ID (used for reproducible random seed in statistical agents)
            reasoning_effort: Reasoning effort level for reasoning models (e.g., 'low', 'medium', 'high' for GPT-5.1)
        """

        # Regular LLM-based solving
        prompt = f"""You are given a puzzle to solve. Solve it and provide ONLY the answer (no explanation).

Puzzle: {puzzle}

Your answer (just the answer in integer, nothing else, no units):"""

        print(f"\n[Agent solving puzzle with {model}]")
        print(f"Puzzle: {puzzle}")

        answer = call_replicate(model, prompt, max_tokens=50, reasoning_effort=reasoning_effort)

        # Clean up the answer (remove extra whitespace, quotes, etc.)
        answer = answer.strip().strip('"').strip("'")

        print(f"Agent's answer: {answer}")

        return answer

    def _extract_decision(self, full_response: str, model: str) -> tuple:
        """Extract succinct decision from agent's structured response

        Returns:
            tuple: (decision, defaulted_to_pass) where defaulted_to_pass is True if response was unclear
        """

        defaulted_to_pass = False
        decision = None

        # Try to parse structured format: "DECISION: ..." and "REASONING: ..."
        lines = full_response.split('\n')
        for line in lines:
            line = line.strip()
            if line.upper().startswith('DECISION:'):
                # Extract decision after "DECISION:"
                decision = line[9:].strip()  # Remove "DECISION:" prefix
                break

        # If we couldn't find structured format, try to extract from unstructured response
        if not decision:
            defaulted_to_pass = True
            # Fallback: try to find keywords in original response
            lower_response = full_response.lower()
            if "verify:" in lower_response or '"verify' in lower_response:
                # Extract target - check for question IDs (Q1, Q2, etc.)
                import re
                question_match = re.search(r'verify:?\s*(q\d+)', full_response, re.IGNORECASE)
                if question_match:
                    question_id = question_match.group(1).upper()
                    decision = f"Verify: {question_id}"
                    defaulted_to_pass = False  # Found a valid action
            elif "volunteer:" in lower_response:
                # Try to extract volunteer line
                for line in lines:
                    if "volunteer:" in line.lower():
                        decision = line.strip()
                        if "Q1" in decision and "Q2" in decision:  # Valid volunteer format
                            defaulted_to_pass = False
                        break

            # If still no valid decision found, default to Pass
            if not decision or defaulted_to_pass:
                decision = "Pass"
                defaulted_to_pass = True

        # Validate decision format
        decision_lower = decision.lower()
        if not (decision_lower.startswith('verify:') or
                decision_lower.startswith('volunteer:') or
                decision_lower == 'pass'):
            # Invalid format, default to Pass
            decision = "Pass"
            defaulted_to_pass = True

        print(f"    → Extracted decision: {decision}" + (f" (defaulted to pass)" if defaulted_to_pass else ""))

        return decision, defaulted_to_pass


"""
Game moderator - manages game state and rules
"""
import random
from typing import Dict, List
from agents.base_agent import Agent

class Moderator:
    def __init__(self, agents: Dict[str, Agent], questions: Dict[str, Dict] = None):
        self.agents = agents
        self.round_number = 0
        self.game_over = False
        self.winner_message = ""

        # Store questions and correct answers: {question_id: {puzzle: str, correct_answer: str}}
        self.questions = questions or {}

        # Capsule system: {(question_id, correct_answer): [(capsule_label, agent_answer), ...]}
        # Example: {("Q1", "60"): [("Q1-A", "60"), ("Q1-B+C", "60"), ("Q1-D", "59")], ("Q2", "78"): [("Q2-B", "78")]}
        self.capsules = {}
    
    def get_alive_agents(self) -> List[str]:
        """Get list of alive agent IDs"""
        return [aid for aid, agent in self.agents.items() if agent.is_alive]
    
    def process_decisions(self, decisions: Dict[str, str]) -> tuple:
        """
        Process all player decisions and determine outcome

        Returns:
            tuple: (announcement message, volunteer_id or None, death_reasons dict)
        """
        
        alive_agents = self.get_alive_agents()
        death_reasons = {}  # Track why each agent died

        if len(alive_agents) == 0:
            self.game_over = True
            return ("💀 Everyone has died. Game Over.", None, death_reasons)
        
        # Check for verifications FIRST (before volunteer check)
        # This is important because agents might mention "volunteer" in their reasoning
        verifications = []
        for agent_id, decision in decisions.items():
            # Look for "Verify:" pattern which is more specific
            if "verify:" in decision.lower() or ('"verify' in decision.lower() and 'verify' in decision.lower()):
                # Extract target question ID (e.g., Q1, Q2)
                target_question = self._extract_target(decision)
                if target_question and target_question in self.questions:
                    verifications.append((agent_id, target_question))

        # Check for volunteers (but exclude agents who are verifying)
        volunteers = []
        verifying_agents = [v[0] for v in verifications]
        for agent_id, decision in decisions.items():
            if agent_id in verifying_agents:
                continue  # Skip agents who are verifying
            # Look for "Volunteer:" pattern or explicit volunteer action
            if "volunteer:" in decision.lower() or decision.strip().lower().startswith("volunteer"):
                volunteers.append((agent_id, decision))

        # Process verifications FIRST (before volunteer) so capsules are updated
        verification_msg = ""
        if verifications:
            verification_msg = self._process_verifications(verifications)

        # Handle multiple volunteers - randomly pick one
        if len(volunteers) > 1:
            print(f"\n⚠️ Multiple volunteers: {[v[0] for v in volunteers]}")
            volunteer, volunteer_decision = random.choice(volunteers)
            print(f"   Randomly selected: {volunteer}")
        elif len(volunteers) == 1:
            volunteer, volunteer_decision = volunteers[0]
        else:
            volunteer = None
            volunteer_decision = None

        # Process volunteer AFTER verifications (so they can use updated capsules)
        if volunteer:
            msg, success = self._check_password(volunteer, volunteer_decision)
            # If volunteer failed, track death reason
            if not success:
                death_reasons[volunteer] = "wrong_password"
            # Combine verification message with volunteer result
            combined_msg = verification_msg + "\n" + msg if verification_msg else msg
            # Return volunteer ID only if they succeeded, otherwise they died
            return (combined_msg, volunteer if success else None, death_reasons)

        # IMPORTANT: Check if there was NO volunteer
        # If no one volunteered, someone must die (even if there were verifications)
        if not volunteer:
            victim = random.choice(alive_agents)
            coins_lost = self.agents[victim].coins  # Save coins before wiping
            self.agents[victim].is_alive = False
            self.agents[victim].coins = 0  # Wipe out coins when player dies
            death_reasons[victim] = "no_volunteer"  # Track death reason

            death_msg = f"\n⚠️ No one volunteered! {victim} was randomly eliminated (coins lost: {coins_lost} → 0).\nRemaining: {self.get_alive_agents()}"

            # Combine verification message with death message
            if verification_msg:
                return (verification_msg + death_msg, None, death_reasons)
            else:
                return (death_msg, None, death_reasons)

        # This line should never be reached, but just in case
        return (verification_msg if verification_msg else "Round complete.", None, death_reasons)
    
    def _check_password(self, volunteer: str, volunteer_decision: str) -> tuple:
        """Check if the volunteer's combined password is correct

        Returns:
            tuple: (announcement message, success boolean)
        """
        # Parse capsule selections from volunteer's decision
        selections = self._parse_capsule_selections(volunteer_decision)

        if not selections:
            # Could not parse selections - treat as failure
            self.agents[volunteer].is_alive = False
            self.agents[volunteer].coins = 0
            return (f"❌ {volunteer} volunteered but selection format was invalid. {volunteer} dies! 💀\nRemaining: {self.get_alive_agents()}", False)

        # Build the password from selected capsules
        password_parts = []
        capsule_labels_selected = []  # Track capsule labels for display
        sorted_question_ids = sorted(self.questions.keys())

        for question_id in sorted_question_ids:
            if question_id not in selections:
                # Missing a selection for this question
                self.agents[volunteer].is_alive = False
                self.agents[volunteer].coins = 0
                return (f"❌ {volunteer} did not select a capsule for {question_id}. {volunteer} dies! 💀\nRemaining: {self.get_alive_agents()}", False)

            selected_capsule = selections[question_id]

            # Find the matching capsule using the helper method
            actual_capsule = self._find_matching_capsule(question_id, selected_capsule)

            if not actual_capsule:
                # Selected capsule doesn't exist
                self.agents[volunteer].is_alive = False
                self.agents[volunteer].coins = 0
                return (f"❌ {volunteer} selected non-existent capsule {selected_capsule}. {volunteer} dies! 💀\nRemaining: {self.get_alive_agents()}", False)

            # Track the capsule label for display
            capsule_labels_selected.append(actual_capsule)

            # Find the capsule in our new structure and get its answer
            capsule_key = None
            for key in self.capsules.keys():
                if key[0] == question_id:
                    capsule_key = key
                    break

            if not capsule_key:
                self.agents[volunteer].is_alive = False
                self.agents[volunteer].coins = 0
                return (f"❌ Internal error: capsule key not found for {question_id}. {volunteer} dies! 💀\nRemaining: {self.get_alive_agents()}", False)

            # Find the answer for the actual capsule
            answer_found = None
            for capsule_label, agent_answer in self.capsules[capsule_key]:
                if capsule_label == actual_capsule:
                    answer_found = agent_answer
                    break

            if answer_found is None:
                self.agents[volunteer].is_alive = False
                self.agents[volunteer].coins = 0
                return (f"❌ Internal error: answer not found for {actual_capsule}. {volunteer} dies! 💀\nRemaining: {self.get_alive_agents()}", False)

            password_parts.append(answer_found)

        # Combine all parts to form the attempted password
        attempted_password = '-'.join(password_parts)
        selected_capsules_display = ', '.join(capsule_labels_selected)

        # Build the correct password from correct answers
        correct_password_parts = []
        for question_id in sorted_question_ids:
            correct_answer = self.questions[question_id]['correct_answer']
            correct_password_parts.append(correct_answer)
        correct_password = '-'.join(correct_password_parts)

        # Check if password is correct
        if attempted_password == correct_password:
            # SUCCESS!
            self.game_over = True
            self.winner_message = f"✅ {volunteer} entered CORRECT password using capsules: {selected_capsules_display}\n🎉 Everyone escapes! 🎉"
            return (self.winner_message, True)
        else:
            # FAILURE
            self.agents[volunteer].is_alive = False
            self.agents[volunteer].coins = 0
            return (f"❌ {volunteer} entered WRONG password using capsules: {selected_capsules_display}\n{volunteer} dies! 💀\nRemaining: {self.get_alive_agents()}", False)



    
    def _extract_target(self, decision: str) -> str:
        """Extract target question ID from decision text (e.g., Q1, Q2, etc.)"""
        import re
        # Look for Q followed by a digit (Q1, Q2, Q3, etc.)
        match = re.search(r'Q(\d+)', decision, re.IGNORECASE)
        if match:
            return f"Q{match.group(1)}"
        return None

    def _parse_capsule_selections(self, decision: str) -> Dict[str, str]:
        """Parse capsule selections from volunteer's decision text"""
        import re

        selections = {}

        # Look for pattern like "Q1-A" or "Q1-A+B" or "Q1: Q1-A" in the decision text
        # Pattern matches Q followed by digit, hyphen, one or more letters (plus or comma separated)
        capsule_pattern = r'(Q\d+)[-:\s]+(Q\d+-[A-Z](?:[+,][A-Z])*)'
        matches = re.findall(capsule_pattern, decision)

        for question_id, capsule_label in matches:
            selections[question_id] = capsule_label

        # Also try simpler pattern without question prefix
        if not selections:
            simple_pattern = r'(Q\d+-[A-Z](?:[+,][A-Z])*)'
            simple_matches = re.findall(simple_pattern, decision)
            for capsule_label in simple_matches:
                question_id = capsule_label.split('-')[0]
                selections[question_id] = capsule_label

        return selections if selections else None

    def _find_matching_capsule(self, question_id: str, selected_capsule: str) -> str:
        """Find a capsule that matches the player's selection, accounting for endorsements.

        When a player selects "Q1-A" but the actual capsule is "Q1-A+B" (because B endorsed A),
        this function will find the match.

        Args:
            question_id: The question ID (e.g., "Q1")
            selected_capsule: The capsule label player selected (e.g., "Q1-A")

        Returns:
            The actual capsule label if found, None otherwise
        """
        if not selected_capsule:
            return None

        # Find the capsule key for this question
        capsule_key = None
        for key in self.capsules.keys():
            if key[0] == question_id:
                capsule_key = key
                break

        if not capsule_key:
            return None

        # Extract the agent ID(s) from the selected capsule (e.g., "Q1-A" -> "A", "Q1-A+B" -> ["A", "B"])
        try:
            import re
            # Format is "Q#-Agent1+Agent2+..." or "Q#-Agent1,Agent2,..."
            agents_part = selected_capsule.split('-')[1] if '-' in selected_capsule else None
            if not agents_part:
                return None
            # Split by either + or , to handle both formats
            selected_agents = set(re.split(r'[+,]', agents_part))
        except (IndexError, AttributeError):
            return None

        # Get the capsule list for this question
        capsule_list = self.capsules[capsule_key]

        # Search for a capsule that contains ANY of the selected agents
        # Prioritize capsules where the first agent matches (original creator)
        for capsule_label, agent_answer in capsule_list:
            try:
                capsule_agents_part = capsule_label.split('-')[1]
                # Split by either + or , to handle both formats
                capsule_agents = re.split(r'[+,]', capsule_agents_part)

                # Check if the first agent (original creator) is in selected_agents
                if capsule_agents[0] in selected_agents:
                    return capsule_label
            except (IndexError, AttributeError):
                continue

        # Fallback: check if any selected agent is in any capsule's agent list
        for capsule_label, agent_answer in capsule_list:
            try:
                capsule_agents_part = capsule_label.split('-')[1]
                # Split by either + or , to handle both formats
                capsule_agents = set(re.split(r'[+,]', capsule_agents_part))

                # Check if there's any overlap
                if selected_agents & capsule_agents:
                    return capsule_label
            except (IndexError, AttributeError):
                continue

        return None

    def _initialize_capsules(self):
        """Initialize capsules for each agent's answer

        Creates capsules in format: {(question_id, correct_answer): [(capsule_label, agent_answer), ...]}
        """
        # Assign question numbers based on sorted agent IDs
        sorted_agents = sorted(self.agents.keys())

        for idx, agent_id in enumerate(sorted_agents):
            question_id = f"Q{idx + 1}"
            capsule_label = f"{question_id}-{agent_id}"

            # Get correct answer for this question
            correct_answer = self.questions[question_id]['correct_answer']
            capsule_key = (question_id, correct_answer)

            # Initialize capsule list if it doesn't exist
            if capsule_key not in self.capsules:
                self.capsules[capsule_key] = []

            # Add this agent's capsule to the list
            agent_answer = self.agents[agent_id].answer
            self.capsules[capsule_key].append((capsule_label, agent_answer))

    def get_capsule_labels(self) -> Dict[str, List[str]]:
        """Get visible capsule labels (without contents) organized by question

        Returns: {"Q1": ["Q1-A", "Q1-B+C"], "Q2": ["Q2-B"], ...}
        """
        labels = {}
        for (question_id, correct_answer), capsule_list in self.capsules.items():
            if question_id not in labels:
                labels[question_id] = []
            # Extract just the capsule labels (first element of each tuple)
            for capsule_label, agent_answer in capsule_list:
                labels[question_id].append(capsule_label)
        return labels

    def get_capsule_display(self) -> str:
        """Get formatted display of all capsule labels (without answers)

        Format: Q1: Q1-A, Q1-B+C, Q1-D
                Q2: Q2-B
        """
        labels_dict = self.get_capsule_labels()
        display = "Available Capsules:\n"
        for question_id in sorted(labels_dict.keys()):
            capsule_labels = labels_dict[question_id]
            display += f"  {question_id}: {', '.join(capsule_labels)}\n"
        return display

    def get_capsule_snapshot(self) -> Dict:
        """Get a deep copy snapshot of current capsule state for logging"""
        import copy
        return copy.deepcopy(self.capsules)

    def _process_verifications(self, verifications: List[tuple]) -> str:
        """Process verification requests: verifier solves target question and creates/endorses capsule"""
        from utils.replicate_helper import call_replicate

        msg = "🔍 Verifications completed:\n"

        for verifier_id, target_question_id in verifications:
            # Deduct 1 coin for verification
            coins_before = self.agents[verifier_id].coins
            self.agents[verifier_id].coins = max(0, self.agents[verifier_id].coins - 1)

            # Check if target question exists
            if target_question_id not in self.questions:
                msg += f"  ❌ {verifier_id} specified invalid question {target_question_id}\n"
                continue

            # Get the target question's puzzle
            target_puzzle = self.questions[target_question_id]['puzzle']

            # Have verifier solve the puzzle
            verifier_model = self.agents[verifier_id].model

            solve_prompt = f"""You are verifying another player's answer. Solve this puzzle and provide ONLY the answer (no explanation).

Puzzle: {target_puzzle}

Your answer (just the answer, nothing else):"""

            print(f"\n[{verifier_id} verifying {target_question_id}'s puzzle with {verifier_model}]")
            print(f"Puzzle: {target_puzzle}")

            verifier_answer = call_replicate(verifier_model, solve_prompt, max_tokens=50)
            verifier_answer = verifier_answer.strip().strip('"').strip("'")

            print(f"{verifier_id}'s answer: {verifier_answer}")

            # Find the capsule key for this question
            capsule_key = None
            for key in self.capsules.keys():
                if key[0] == target_question_id:
                    capsule_key = key
                    break

            if not capsule_key:
                msg += f"  ❌ {verifier_id} - no capsules found for {target_question_id}\n"
                continue

            # Check if verifier's answer matches any existing capsule for this question
            from utils.answer_checker import check_arithmetic_answer
            matched_index = None
            capsule_list = self.capsules[capsule_key]

            for idx, (capsule_label, agent_answer) in enumerate(capsule_list):
                answers_match = check_arithmetic_answer(agent_answer, verifier_answer)
                if answers_match:
                    matched_index = idx
                    break

            if matched_index is not None:
                # Endorse existing capsule by updating its label
                old_label, answer = capsule_list[matched_index]

                # Check if verifier already endorsed this capsule
                if verifier_id in old_label:
                    msg += f"  ℹ️ {verifier_id} already endorsed {old_label}\n"
                    continue

                # Extract existing agents from label (e.g., "Q1-A+B" -> ["A", "B"])
                agents_part = old_label.split('-')[1]
                existing_agents = agents_part.split('+')

                # Add verifier to the list
                existing_agents.append(verifier_id)
                new_label = f"{target_question_id}-{'+'.join(existing_agents)}"

                # Replace the capsule tuple with updated label
                capsule_list[matched_index] = (new_label, answer)

                msg += f"  ✅ {verifier_id} agrees with {old_label} → now {new_label} (-1 coin: {coins_before}→{self.agents[verifier_id].coins})\n"
            else:
                # Create new capsule with verifier's answer
                new_label = f"{target_question_id}-{verifier_id}"
                capsule_list.append((new_label, verifier_answer))

                msg += f"  🆕 {verifier_id} created new capsule {new_label} (conflicting answer) (-1 coin: {coins_before}→{self.agents[verifier_id].coins})\n"

        return msg

    def conduct_post_game_review(self) -> str:
        """
        Reveal correct answers and compare them with each agent's answer.
        Returns a formatted review message.
        """
        review_lines = []
        review_lines.append("\n" + "=" * 80)
        review_lines.append("📋 POST-GAME REVIEW: Answer Revelation")
        review_lines.append("=" * 80)

        # Map questions to agents (Q1->A, Q2->B, etc.)
        sorted_agent_ids = sorted(self.agents.keys())

        for idx, agent_id in enumerate(sorted_agent_ids):
            question_id = f"Q{idx + 1}"

            # Get the correct answer
            correct_answer = self.questions[question_id]['correct_answer']
            puzzle = self.questions[question_id]['puzzle']

            # Get the agent's answer
            agent_answer = self.agents[agent_id].answer

            # Compare answers using algorithmic checking
            from utils.answer_checker import check_arithmetic_answer
            is_correct = check_arithmetic_answer(correct_answer, agent_answer)

            status = "✅ CORRECT" if is_correct else "❌ WRONG"

            # review_lines.append(f"\n{question_id} - Assigned to {agent_id}:")
            # review_lines.append(f"  Puzzle: {puzzle}")
            # review_lines.append(f"  Correct Answer: {correct_answer}")
            # review_lines.append(f"  {agent_id}'s Answer: {agent_answer}")
            # review_lines.append(f"  Result: {status}")

        review_lines.append("\n" + "=" * 80)

        review_message = "\n".join(review_lines)
        print(review_message)

        return review_message
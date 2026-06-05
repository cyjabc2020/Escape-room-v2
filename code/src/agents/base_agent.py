"""
Base agent class for game players

Memory System Architecture:
- Short-term Memory: Recent game events (last_n, current_game_only, max_allowed)
- Long-term Memory: Cross-game reflections (evolving)
"""
from utils.replicate_helper import call_replicate

class Agent:
    def __init__(self, agent_id: str, model: str, puzzle: str, answer: str = None,
                 memory_mode: str = "last_n", memory_window: int = 5,
                 max_context_tokens: int = 1000, max_output_tokens: int = 300,
                 initial_coins: int = 5, reflection_mode: str = "none",
                 reasoning_effort: str = None, correct_answer: str = None,
                 dummy_config: dict = None):
        """
        Initialize agent with dual memory system: short-term and long-term

        Args:
            agent_id: Unique identifier for agent
            model: LLM model to use
            puzzle: Agent's puzzle
            answer: Agent's answer to puzzle (may be set later)
            memory_mode: Short-term memory mode - "none", "last_n", "current_game_only", or "max_allowed"
            memory_window: Number of recent memories to keep (for "last_n" mode), or max events limit (for "max_allowed" mode)
            max_context_tokens: Maximum tokens for context (for "max_allowed" modes, default 3000 for max_allowed)
            max_output_tokens: Maximum tokens for model output
            initial_coins: Starting number of coins
            reflection_mode: Long-term memory mode - "none" or "evolving"
            reasoning_effort: Reasoning effort level for reasoning models (e.g., 'none', 'low', 'medium', 'high' for GPT-5.1)
            correct_answer: The correct answer to the puzzle
            dummy_config: Configuration for dummy agents (dict with 'correctness_list' key containing bool values per game)
        """
        self.agent_id = agent_id
        self.model = model
        self.puzzle = puzzle
        self.answer = answer
        self.correct_answer = correct_answer
        self.is_alive = True
        self.coins = initial_coins
        self.reasoning_effort = reasoning_effort
        self.dummy_config = dummy_config if dummy_config else {}

        # ========== SHORT-TERM MEMORY ==========
        # Stores recent game events and decisions
        self.memory = []  # Full memory history
        self.memory_mode = memory_mode  # "none", "last_n", "current_game_only", "max_allowed"
        self.memory_window = memory_window
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens

        # For current_game_only mode
        self.current_game_memory = []  # Memory for current game only

        # ========== LONG-TERM MEMORY ==========
        # Stores reflections and learnings across games
        self.reflection_mode = reflection_mode  # "none", "evolving"
        self.evolving_reflection = ""  # Single reflection that evolves across games (for "evolving" mode)

    # ========== SHORT-TERM MEMORY METHODS ==========

    def add_memory(self, message: str):
        """Add event to short-term memory"""
        self.memory.append(message)

        # For current_game_only mode, also add to current game memory
        if self.memory_mode == "current_game_only":
            self.current_game_memory.append(message)

    def reset_current_game_memory(self):
        """Reset short-term memory for current game (called when a new game starts)"""
        if self.memory_mode == "current_game_only":
            self.current_game_memory = []

    def clear_memory(self):
        """Clear all short-term memories (useful between games)"""
        self.memory = []

    # ========== LONG-TERM MEMORY METHODS ==========
    def _get_agent_reflection(self) -> str:
        """Get post-game reflection from an agent

        Returns:
            str: Agent's reflection on the game
        """
        if self.reflection_mode == "none":
            reflection = ""
        elif self.model == "dummy":
            reflection = "Dummy has no reflection."
        elif self.reflection_mode == "evolving":
            context = self.get_context()

            prompt = f"""Based on the following short-term memory and previous reflection, create a concise reflection (max 300 words) about:
    - What strategies worked or didn't work
    - What you learned about the game mechanics
    - What you might do differently next time

    Memory and Game Outcome:
    {context}

    Reflection:"""

            # Get reflection from agent
            reflection = call_replicate(self.model, prompt, reasoning_effort=self.reasoning_effort)
            self.evolving_reflection = reflection

        return reflection
    
    def get_context(self) -> str:
        """Get agent's full context combining short-term and long-term memory

        Returns a context string with two sections:
        1. Short-term memory: Recent game events
        2. Long-term memory: Cross-game reflections
        """
        context_parts = []

        # 1. Add short-term memory (recent events)
        short_term = self._get_short_term_memory()
        if short_term:
            context_parts.append(short_term)

        # 2. Add long-term memory (reflections)
        long_term = self._get_long_term_memory()
        if long_term:
            context_parts.append(long_term)

        return "\n\n".join(context_parts) if context_parts else ""

    def _get_short_term_memory(self) -> str:
        """Get short-term memory context (recent game events)

        Short-term memory modes:
        - none: No short-term memory
        - last_n: Last N events across all games
        - current_game_only: Only events from current game
        - max_allowed: Keep as many recent events as possible within token limit (default 5000)
        """
        if self.memory_mode == "none":
            return ""

        elif self.memory_mode == "last_n":
            # Return last N memories
            recent_memories = self.memory[-self.memory_window:]
            return "\n".join(recent_memories)

        elif self.memory_mode == "current_game_only":
            # Return only memories from current game
            return "\n".join(self.current_game_memory)

        elif self.memory_mode == "max_allowed":
            # Keep as many recent events as possible within token limit
            # Uses max_context_tokens (default 3000) as the limit
            # Also respects memory_window as the maximum number of events
            max_tokens = self.max_context_tokens if self.max_context_tokens else 3000
            max_events = self.memory_window if self.memory_window > 0 else len(self.memory)

            # Start from most recent and work backwards
            selected_memories = []
            total_chars = 0
            # Rough estimate: 1 token ≈ 4 characters
            max_chars = max_tokens * 4

            # Get memories in reverse order (most recent first)
            for i, mem in enumerate(reversed(self.memory)):
                if i >= max_events:
                    break
                mem_chars = len(mem) + 1  # +1 for newline
                if total_chars + mem_chars > max_chars:
                    break
                selected_memories.append(mem)
                total_chars += mem_chars

            # Reverse back to chronological order
            selected_memories.reverse()
            return "\n".join(selected_memories)

        else:
            # Default fallback
            return "\n".join(self.memory[-5:])

    def _get_long_term_memory(self) -> str:
        """Get long-term memory context (cross-game reflections)

        Long-term memory modes:
        - none: No long-term memory
        - evolving: Single reflection that evolves across games
        """
        if self.reflection_mode == "none":
            return ""

        elif self.reflection_mode == "evolving":
            # Return the single evolving reflection
            if self.evolving_reflection:
                return f"[Previous Games Reflection]:\n{self.evolving_reflection}"
            return ""

        return ""

    def __repr__(self):
        return f"Agent({self.agent_id}, alive={self.is_alive}, STM={self.memory_mode}, LTM={self.reflection_mode})"
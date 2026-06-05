"""
Question Bank Manager for loading and selecting questions from CSV files
"""
import csv
import random
from pathlib import Path
from typing import List, Dict, Optional


class Question:
    """Represents a single question"""
    def __init__(self, question: str, answer: str):
        self.question = question
        self.answer = answer

    def to_dict(self):
        return {
            'question': self.question,
            'answer': self.answer,
        }


class QuestionBank:
    """Manages question banks loaded from CSV files"""

    def __init__(self, question_banks_dir: str = "question_banks"):
        self.question_banks_dir = Path(question_banks_dir)
        self.questions: List[Question] = []
        self.loaded_files: List[str] = []
        self.used_questions: set = set()  # Track used questions to avoid duplicates

    def load_question_set(self, csv_filename: str) -> int:
        """Load questions from a CSV file

        Args:
            csv_filename: Name of the CSV file (e.g., "basic_arithmetic.csv")

        Returns:
            Number of questions loaded
        """
        filepath = self.question_banks_dir / csv_filename

        if not filepath.exists():
            raise FileNotFoundError(f"Question set not found: {filepath}")

        questions_loaded = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:

                question = Question(
                    question=row['question'].strip(),
                    answer=row['answer'].strip(),
                )
                self.questions.append(question)
                questions_loaded += 1

        self.loaded_files.append(csv_filename)
        return questions_loaded

    def load_multiple_sets(self, csv_filenames: List[str]) -> int:
        """Load questions from multiple CSV files

        Args:
            csv_filenames: List of CSV filenames to load

        Returns:
            Total number of questions loaded across all files
        """
        total_loaded = 0
        for filename in csv_filenames:
            loaded = self.load_question_set(filename)
            total_loaded += loaded
        return total_loaded

    def get_random_questions(self, n: int, seed: Optional[int] = None) -> List[Question]:
        """Get N random unique questions from the loaded question bank

        Args:
            n: Number of questions to select
            seed: Optional random seed for reproducibility

        Returns:
            List of randomly selected Question objects

        Raises:
            ValueError: If n is greater than available questions
        """
        # Filter out already used questions
        available_questions = [q for q in self.questions if q.question not in self.used_questions]

        if n > len(available_questions):
            raise ValueError(
                f"Requested {n} questions but only {len(available_questions)} unused questions available "
                f"(total: {len(self.questions)}, used: {len(self.used_questions)}). "
                f"Load more question sets or reduce the number of players."
            )

        # Set random seed if provided
        if seed is not None:
            random.seed(seed)

        # Select n unique random questions from available pool
        selected = random.sample(available_questions, n)

        # Mark these questions as used
        for q in selected:
            self.used_questions.add(q.question)

        return selected

    def get_stats(self) -> Dict:
        """Get statistics about loaded questions"""
        return {
            'total_questions': len(self.questions),
            'loaded_files': self.loaded_files,
        }

    def list_available_question_sets(self) -> List[str]:
        """List all available CSV files in the question_banks directory"""
        if not self.question_banks_dir.exists():
            return []

        csv_files = [f.name for f in self.question_banks_dir.glob("*.csv")]
        return sorted(csv_files)

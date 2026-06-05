"""
Helper functions for checking answer correctness algorithmically
"""
import re


def normalize_number(text: str) -> float:
    """
    Extract and normalize a number from text.

    Args:
        text: String potentially containing a number

    Returns:
        Normalized number as float, or None if no number found
    """
    if not text:
        return None

    # Remove common text patterns
    text = text.strip().lower()

    # Remove dollar signs, currency symbols
    text = re.sub(r'[\$£€¥]', '', text)

    # Remove commas from numbers
    text = re.sub(r',', '', text)

    # Handle word numbers (basic cases)
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
        'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
        'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
        'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
        'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
        'hundred': 100, 'thousand': 1000, 'million': 1000000
    }

    for word, num in word_to_num.items():
        if word in text:
            text = text.replace(word, str(num))

    # Extract number (int or float)
    # Try to find a number pattern
    number_pattern = r'-?\d+\.?\d*'
    matches = re.findall(number_pattern, text)

    if not matches:
        return None

    # Take the first number found
    try:
        return float(matches[0])
    except (ValueError, IndexError):
        return None


def check_arithmetic_answer(correct_answer: str, submitted_answer: str, tolerance: float = 0.01) -> bool:
    """
    Check if two arithmetic answers are equivalent.

    This function normalizes both answers by extracting numerical values
    and comparing them within a tolerance.

    Args:
        correct_answer: The correct answer
        submitted_answer: The submitted answer to check
        tolerance: Acceptable difference for floating point comparison (default: 0.01)

    Returns:
        True if answers are equivalent, False otherwise
    """
    # Normalize both answers
    correct_num = normalize_number(correct_answer)
    submitted_num = normalize_number(submitted_answer)

    # If either couldn't be parsed as a number, they're not equal
    if correct_num is None or submitted_num is None:
        return False

    # Compare within tolerance
    return abs(correct_num - submitted_num) <= tolerance
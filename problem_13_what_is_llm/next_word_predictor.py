"""Simple frequency-based next-word predictor for customer-service text."""

import re
from collections import Counter, defaultdict
from pathlib import Path


# Change this value to use a different number of words in the input phrase.
N = 3
TRAINING_FILE = Path(__file__).with_name("training_data.txt")
DOMAIN = "Branch, Contact Center & Service Request Fulfillment Agent"


def load_training_data(file_path=TRAINING_FILE):
    """Read the training text from a plain-text file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Training file not found: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("The training file is empty.")
    return text


def tokenize(text):
    """Convert text to lowercase words and remove unnecessary punctuation."""
    # Keeping only words and numbers makes training and user input consistent.
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def build_frequency_table(tokens, phrase_length=N):
    """Count which words follow each phrase in the training text."""
    if phrase_length < 1:
        raise ValueError("Phrase length must be at least 1.")

    table = defaultdict(Counter)
    for index in range(len(tokens) - phrase_length):
        phrase = tuple(tokens[index:index + phrase_length])
        next_word = tokens[index + phrase_length]
        table[phrase][next_word] += 1
    return table


class NextWordPredictor:
    """Predict the most frequent word after a phrase."""

    def __init__(self, frequency_table, phrase_length=N):
        self.frequency_table = frequency_table
        self.phrase_length = phrase_length

    def predict_next_word(self, phrase):
        """Return the prediction for the last N words of the input phrase."""
        phrase_tokens = tokenize(phrase)
        if len(phrase_tokens) < self.phrase_length:
            return None, 0.0

        # If the user enters more than N words, use the most recent N words.
        context = tuple(phrase_tokens[-self.phrase_length:])
        counts = self.frequency_table.get(context)
        if not counts:
            return None, 0.0

        # Sort by highest count first, then alphabetically for deterministic ties.
        predicted_word, frequency = min(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
        total_occurrences = sum(counts.values())
        confidence = frequency / total_occurrences * 100
        return predicted_word, confidence


def run_cli(predictor):
    """Run an interactive console loop until the user types exit or quit."""
    print("\nSimple Next-Word Predictor")
    print(f"Domain: {DOMAIN}")
    print(f"Phrase length: {predictor.phrase_length}")
    print(
        f"You may enter any number of words; the last "
        f"{predictor.phrase_length} words will be used for prediction."
    )
    print("Type 'exit' or 'quit' to stop.")

    while True:
        phrase = input("\nEnter a phrase: ").strip()
        if phrase.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not phrase:
            print("Please enter a phrase.")
            continue

        phrase_tokens = tokenize(phrase)
        if len(phrase_tokens) < predictor.phrase_length:
            print(
                f"Please enter at least {predictor.phrase_length} words. "
                f"The last {predictor.phrase_length} words will be used."
            )
            continue

        predicted_word, confidence = predictor.predict_next_word(phrase)
        if predicted_word is None:
            print("No prediction available for this phrase.")
            continue

        print(f"Predicted word: {predicted_word}")
        print(f"Confidence: {confidence:.2f}%")


def main():
    """Load data, train the frequency table, and start the predictor."""
    try:
        print("Loading training data...")
        training_text = load_training_data()
        tokens = tokenize(training_text)
        if len(tokens) <= N:
            raise ValueError("The training data is too short for this phrase length.")

        print("Building frequency table...")
        frequency_table = build_frequency_table(tokens, N)
        predictor = NextWordPredictor(frequency_table, N)
        print("Training completed.")
        run_cli(predictor)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()

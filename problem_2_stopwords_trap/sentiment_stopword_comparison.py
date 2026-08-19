"""Show how removing common words can change sentiment results."""

# NLTK provides tools for working with human language.
import nltk
# stopwords supplies common words such as "the", "is", and "not".
from nltk.corpus import stopwords
# VADER estimates whether text sounds positive, negative, or neutral.
from nltk.sentiment import SentimentIntensityAnalyzer


# Download the vocabulary used by VADER and the English stop-word list.
# These downloads are safe to repeat when the program is run again.
nltk.download("vader_lexicon", quiet=True)
nltk.download("stopwords", quiet=True)


# These examples contain negation words that can change the meaning of a sentence.
SENTENCES = [
    "The branch service is not good.",
    "I have never had a better experience with the contact center.",
    "The service request was not bad at all.",
    "There is no way this service request will work.",
    "I am not disappointed with the results.",
]


def polarity(score):
    """Convert a numerical score into positive, negative, or neutral."""
    # A score above zero is positive, below zero is negative, and exactly zero
    # means the text is neutral according to this simple comparison.
    if score > 0:
        return "positive"
    if score < 0:
        return "negative"
    return "neutral"


def compare_polarity(original_score, filtered_score):
    """Check whether removing stop words changed the sentiment direction."""
    # First convert both numerical scores into easy-to-read labels.
    original_polarity = polarity(original_score)
    filtered_polarity = polarity(filtered_score)

    # If both labels match, stop-word removal did not change the direction.
    if original_polarity == filtered_polarity:
        return "SAME"
    # A positive-to-negative or negative-to-positive change is a true flip.
    if {original_polarity, filtered_polarity} == {"positive", "negative"}:
        return "FLIPPED"
    # This covers a change involving neutral sentiment.
    return "CHANGED_FROM_NEUTRAL"


def print_table(results):
    """Print the five comparisons in aligned columns."""
    # These headings describe what the user will see in each column.
    headers = (
        "#",
        "Original Sentence",
        "Filtered Sentence",
        "Result",
    )

    # Convert the internal results into printable table rows.
    rows = [
        (
            str(index),
            result["original"],
            result["filtered"],
            result["comparison"],
        )
        for index, result in enumerate(results, start=1)
    ]
    # Calculate each column's width so that long sentences still align neatly.
    widths = [
        max(len(headers[index]), max(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    separator = "-+-".join("-" * width for width in widths)

    # Print the title, headings, divider, and each result row.
    print("\nSentiment Before and After Stop-Word Removal")
    print("=" * len(separator))
    print(" | ".join(
        f"{header:<{widths[index]}}"
        for index, header in enumerate(headers)
    ))
    print(separator)
    for row in rows:
        print(" | ".join(
            f"{value:<{widths[index]}}"
            for index, value in enumerate(row)
        ))


def main():
    # Create the VADER analyzer and load the English stop-word collection.
    analyzer = SentimentIntensityAnalyzer()
    english_stop_words = set(stopwords.words("english"))
    results = []

    # Analyze each example twice: first as written, then after filtering.
    for sentence in SENTENCES:
        # Calculate VADER sentiment on the untouched original sentence.
        original_score = analyzer.polarity_scores(sentence)["compound"]

        # Remove English stop words while preserving the remaining word order.
        # This intentionally removes words like "not", "never", and "no" so
        # the program can demonstrate why careless filtering is risky.
        tokens = sentence.split()
        filtered_sentence = " ".join(
            token for token in tokens
            if token.lower().strip(".,!?;:") not in english_stop_words
        )

        # Calculate VADER sentiment again after stop-word removal.
        filtered_score = analyzer.polarity_scores(filtered_sentence)["compound"]

        # Compare the two score signs programmatically, including neutral cases.
        results.append({
            "original": sentence,
            "filtered": filtered_sentence,
            "original_score": original_score,
            "filtered_score": filtered_score,
            "comparison": compare_polarity(original_score, filtered_score),
        })

    # Display the required five-row comparison table.
    print_table(results)

    # Separate changed and unchanged cases for the written analysis below.
    changed = [
        result for result in results if result["comparison"] != "SAME"
    ]
    unchanged = [
        result for result in results if result["comparison"] == "SAME"
    ]

    # Explain what happened and why this matters for customer-service text.
    print("\nAnalysis")
    print("=" * 60)
    print(f"Changed after stop-word removal: {len(changed)} sentence(s).")
    for result in changed:
        print(f"- {result['original']} [{result['comparison']}]")

    print(f"Remained the same: {len(unchanged)} sentence(s).")
    for result in unchanged:
        print(f"- {result['original']}")

    print(
        "\nRemoving words such as 'not', 'never', and 'no' can be dangerous "
        "because they reverse or strongly modify the meaning of a statement."
    )
    print(
        "For example, 'the service is not working' does not mean the same "
        "thing as 'the service working'. In a Branch, Contact Center, and "
        "Service Request Fulfillment Agent, removing negations can cause "
        "incorrect satisfaction, escalation, and request-priority decisions."
    )
    print(
        "Recommendation: do not apply general stop-word removal before "
        "sentiment analysis in this domain. Preserve negation words, or use "
        "a domain-aware filtering strategy that explicitly protects them."
    )


if __name__ == "__main__":
    # Start the program only when this file is run directly.
    main()

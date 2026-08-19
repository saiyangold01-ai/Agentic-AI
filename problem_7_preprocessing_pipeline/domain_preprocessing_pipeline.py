"""Compare four NLP preprocessing configurations for customer-service text."""

# re finds and removes patterns such as URLs, email addresses, and punctuation.
import re
# Iterable and list[str] make the metric function easier to understand and check.
from collections.abc import Iterable

# NLTK provides the language-processing tools used in this example.
import nltk
# stopwords contains common words such as "the" and "is".
from nltk.corpus import stopwords
# The lemmatizer changes words to useful dictionary base forms.
from nltk.stem import WordNetLemmatizer
# This turns a sentence into separate word tokens.
from nltk.tokenize import word_tokenize


# Download the NLTK resources needed by tokenization, stopword removal,
# and lemmatization. These downloads are safe to repeat.
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# Exactly three messy examples from branch, contact-center, and fulfillment work.
# They intentionally contain URLs, email addresses, capitals, punctuation, and
# extra spaces so the cleaning process can be observed.
TEXTS = [
    """  BRANCH UPDATE!!! Please visit https://bank.example.com/branch  or email
    HelpDesk@Bank.Example.com about my PENDING address-change REQUEST... it is NOT
    showing in the branch system.  Customer asks: can you check? """,
    """Contact center alert: FAILED card transaction at 09:45; amount was debited,
    but no reversal yet!!! Reply to support@bank.example.com -- I NEVER received
    a clear update.  Ticket #CC-204 must be checked, please. """,
    """Service-request fulfillment: the document upload was marked RESOLVED at
    https://portal.example.com/requests/778, but the requested statement is NOT
    available.  Please review the pending request, missing document, and next step.
    """,
]


# Negation words are normally NLTK stopwords, but they can completely change
# meaning in customer-service text. We always preserve them.
PROTECTED_WORDS = {"not", "no", "never"}

CONFIGURATIONS = (
    ("Stopwords ON, Lemmatization ON", True, True),
    ("Stopwords ON, Lemmatization OFF", True, False),
    ("Stopwords OFF, Lemmatization ON", False, True),
    ("Stopwords OFF, Lemmatization OFF", False, False),
)


def preprocess(text, remove_stopwords=True, lemmatize=True):
    """Clean one message and return its useful normalized words."""
    # Replace URLs and email addresses with spaces so their punctuation does
    # not create misleading tokens. Their actual values are not needed here.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", text)

    # Lowercase makes words such as "Branch" and "branch" equivalent.
    text = text.lower()
    # Keep letters, numbers, and spaces; remove punctuation and extra symbols.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Convert the cleaned sentence into individual word tokens.
    tokens = word_tokenize(text)
    # Load common English words into a set so membership checks are fast.
    english_stopwords = set(stopwords.words("english"))
    # Create the dictionary-based tool used when lemmatization is enabled.
    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for token in tokens:
        # Keep domain-critical negations even when stopword removal is enabled.
        # Remove ordinary filler words, but keep negations such as "not".
        if remove_stopwords and token in english_stopwords and token not in PROTECTED_WORDS:
            continue
        # Lemmatization groups related forms, such as "requests" and "request".
        if lemmatize:
            token = lemmatizer.lemmatize(token, pos="v")
            token = lemmatizer.lemmatize(token, pos="n")
        cleaned_tokens.append(token)

    return cleaned_tokens


def calculate_metrics(processed_texts: Iterable[list[str]]) -> dict[str, float]:
    """Calculate simple measurements for one preprocessing configuration."""
    # Flatten all message token lists into one list for counting.
    text_lists = list(processed_texts)
    tokens = [token for text in text_lists for token in text]
    # Recalculate the unfiltered token count to use as the comparison baseline.
    original_tokens = sum(
        len(preprocess(text, remove_stopwords=False, lemmatize=False))
        for text in TEXTS
    )
    return {
        "tokens_remaining": float(len(tokens)),
        "vocabulary_size": float(len(set(tokens))),
        "average_tokens_per_text": len(tokens) / len(text_lists),
        "percentage_tokens_removed": (
            (1 - len(tokens) / original_tokens) * 100
            if original_tokens
            else 0.0
        ),
    }


def display_configuration(name, processed_texts, metrics):
    """Show original text, cleaned text, and measurements for one setting."""
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)
    # Print each original message beside its cleaned version.
    for index, (original, tokens) in enumerate(zip(TEXTS, processed_texts), start=1):
        print(f"\nText {index} - Original:\n{original.strip()}")
        print(f"Text {index} - Processed:\n{' '.join(tokens)}")

    print("\nMetrics:")
    print(f"- Tokens remaining: {int(metrics['tokens_remaining'])}")
    print(f"- Vocabulary size: {int(metrics['vocabulary_size'])}")
    print(f"- Average tokens per text: {metrics['average_tokens_per_text']:.2f}")
    print(f"- Percentage of tokens removed: {metrics['percentage_tokens_removed']:.2f}%")


def main():
    """Run every message through all four preprocessing choices."""
    print("DOMAIN NLP PREPROCESSING COMPARISON")
    print("Branch, Contact Center & Service Request Fulfillment Agent")
    print("\nProtected negation words: not, no, never")

    # Store metrics for every configuration so they can be compared if needed.
    all_metrics = {}
    for name, remove_stopwords, lemmatize in CONFIGURATIONS:
        # Use the same preprocess function each time, changing only its options.
        processed_texts = [
            preprocess(
                text,
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize,
            )
            for text in TEXTS
        ]
        metrics = calculate_metrics(processed_texts)
        all_metrics[name] = metrics
        display_configuration(name, processed_texts, metrics)

    # Explain the trade-off and state the domain-specific recommendation.
    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)
    print(
        "The metrics show the expected trade-off: stopword removal produces "
        "shorter text, while lemmatization groups related forms such as "
        "requests/request and resolved/resolve without removing important words."
    )
    print(
        "For this domain, preserving negation is more important than achieving "
        "the smallest token count. The stopword-enabled configurations are "
        "therefore useful for search or topic analysis only because not, no, "
        "and never are explicitly protected."
    )
    print(
        "\nBest Configuration: Stopwords ON, Lemmatization ON\n\n"
        "Reason: It reduces common filler words and produces consistent base forms, "
        "which improves matching across branch, contact-center, and fulfillment "
        "terms. The domain-specific protection rule keeps not, no, and never, "
        "so negative customer statements retain their original meaning."
    )


if __name__ == "__main__":
    # Run main only when this file is executed directly.
    main()

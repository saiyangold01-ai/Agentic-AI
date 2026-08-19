"""Compare two ways of reducing words to simpler forms."""

# NLTK is a Python toolkit for working with human language.
import nltk
# The stemmer cuts word endings; the lemmatizer finds dictionary forms.
from nltk.stem import PorterStemmer, WordNetLemmatizer


# WordNet is the dictionary used by the lemmatizer. Downloading it allows the
# program to run on a fresh computer. Repeating the download is harmless.
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# These are the banking and customer-service words used in the comparison.
WORDS = [
    "customers",
    "requests",
    "services",
    "branches",
    "agents",
    "contacted",
    "contacting",
    "resolved",
    "resolving",
    "submitted",
    "submitting",
    "assigned",
    "pending",
    "inquiries",
    "complaints",
]

# WordNet uses short labels for parts of speech:
# n = noun, v = verb, and a = adjective.
# Giving this information helps the lemmatizer choose the correct base word.
PARTS_OF_SPEECH = {
    "customers": "n",
    "requests": "n",
    "services": "n",
    "branches": "n",
    "agents": "n",
    "contacted": "v",
    "contacting": "v",
    "resolved": "v",
    "resolving": "v",
    "submitted": "v",
    "submitting": "v",
    "assigned": "v",
    "pending": "a",
    "inquiries": "n",
    "complaints": "n",
}


def print_results(results):
    """Print the three word forms in a readable table."""
    # These headings explain what each column contains.
    headers = ("Original Word", "Stemmed Word", "Lemmatized Word")
    # Calculate a wide enough column for both the heading and its values.
    widths = [
        max(len(headers[index]), max(len(row[index]) for row in results))
        for index in range(3)
    ]
    separator = "-+-".join("-" * width for width in widths)

    # Print the title, column names, and a separator line.
    print("\nStemming vs. Lemmatization")
    print(f"{headers[0]:<{widths[0]}} | {headers[1]:<{widths[1]}} | "
          f"{headers[2]:<{widths[2]}}")
    print(separator)

    # Print one row for each word.
    for row in results:
        print(f"{row[0]:<{widths[0]}} | {row[1]:<{widths[1]}} | "
              f"{row[2]:<{widths[2]}}")


def main():
    # Create the two NLP tools that will be compared.
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Process every word with both tools and keep the results together.
    results = [
        (
            word,
            stemmer.stem(word),
            lemmatizer.lemmatize(word, pos=PARTS_OF_SPEECH[word]),
        )
        for word in WORDS
    ]

    # Display the main comparison table.
    print_results(results)

    # Display only cases where stemming and lemmatization differ.
    print("\nExamples where stemming produces an incomplete form")
    print("-" * 55)
    for original, stemmed, lemmatized in results:
        if stemmed != lemmatized:
            print(f"- {original}: stem='{stemmed}', lemma='{lemmatized}'")

    # Explain the difference in simple language.
    print("\nExplanation")
    print("-" * 55)
    print(
        "Stemming removes word endings using simple rules. It is fast, "
        "but may produce incomplete or non-dictionary words."
    )
    print(
        "Lemmatization uses vocabulary and part-of-speech information to "
        "produce a meaningful dictionary base form."
    )

    # Explain which method is more useful for this customer-service domain.
    print("\nDomain Recommendation")
    print("-" * 55)
    print(
        "Lemmatization is more suitable for a Branch, Contact Center, "
        "and Service Request Fulfillment Agent because meaningful base "
        "words improve intent detection, search, request routing, and "
        "customer-service analytics."
    )

    # Give the final recommendation.
    print("\nConclusion")
    print("-" * 55)
    print(
        "Lemmatization is better for this domain because it preserves "
        "interpretable dictionary words and supports more accurate NLP."
    )


if __name__ == "__main__":
    # Run the program only when this file is opened directly with Python.
    main()

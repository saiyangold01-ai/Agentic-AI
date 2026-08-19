"""Find words that are especially important in different service documents."""

# TfidfVectorizer turns documents into numbers so we can compare their words.
from sklearn.feature_extraction.text import TfidfVectorizer


# These five example documents describe different customer-service activities.
# They are deliberately different so TF-IDF can find words that distinguish them.
DOCUMENTS = [
    (
        "Document 1 - Branch Operations",
        ""
        "The branch manager reviewed teller queues, cash replenishment, "
        "vault balancing, account opening, branch appointments, and "
        "customer identity verification during morning operations."
    ),
    (
        "Document 2 - Contact Center Interaction",
        ""
        "The contact center agent handled an inbound callback about a failed "
        "card payment, verified the customer profile, recorded the inquiry, "
        "and scheduled a follow up call with the customer."
    ),
    (
        "Document 3 - Complaint Escalation",
        ""
        "The customer complaint involved repeated billing errors and an "
        "unresolved fee dispute. The case required escalation to a supervisor, "
        "priority review, and a formal apology from the service team."
    ),
    (
        "Document 4 - Service Request Fulfillment",
        ""
        "The service request fulfillment team processed a cheque book order, "
        "updated the delivery address, assigned the request to operations, "
        "and confirmed dispatch after document verification."
    ),
    (
        "Document 5 - SLA Case Resolution",
        ""
        "The case management team monitored SLA deadlines, tracked pending "
        "tasks, notified the assigned owner, completed root cause analysis, "
        "and recorded the final resolution before closure."
    ),
]


# These explanations describe the business meaning of possible important words.
# If a word is not listed, the program uses a general explanation instead.
WORD_EXPLANATIONS = {
    "teller": "It identifies frontline branch-cash operations and is concentrated in the branch document.",
    "vault": "It points specifically to branch cash-control and balancing activity.",
    "replenishment": "It describes a distinctive branch operation involving restoration of cash supplies.",
    "callback": "It identifies customer follow-up activity typical of contact-center interactions.",
    "inbound": "It distinguishes incoming contact-center traffic from other service activities.",
    "card": "It narrows the contact-center issue to a payment-card problem.",
    "billing": "It identifies a complaint involving charges or invoicing rather than general service work.",
    "escalation": "It signals that a complaint requires higher-level intervention.",
    "supervisor": "It identifies the management role involved in reviewing an escalated case.",
    "cheque": "It identifies a specific fulfillment request for a cheque book.",
    "dispatch": "It indicates the delivery stage of a fulfilled service request.",
    "fulfillment": "It describes the operational completion of a customer service request.",
    "sla": "It identifies service-level deadline tracking and is strongly associated with case management.",
    "deadline": "It highlights time-bound service obligations in case management.",
    "resolution": "It identifies the final outcome recorded before a case is closed.",
}


def explain_word(word, document_name, score):
    """Explain why a word is important and show its strongest score."""
    # Use a specific business explanation when one is available.
    explanation = WORD_EXPLANATIONS.get(
        word,
        "It is concentrated in this document and is less common in the other documents."
    )
    return (
        f"{explanation} Highest TF-IDF: {score:.4f} in {document_name}."
    )


def main():
    # Separate the document names from the document text so each can be used
    # for its own purpose later in the program.
    document_names = [name for name, _ in DOCUMENTS]
    document_texts = [text for _, text in DOCUMENTS]

    # TfidfVectorizer performs several preparation steps automatically:
    # lowercase conversion, punctuation handling, word splitting, and removal
    # of common English words such as "the", "and", and "is".
    vectorizer = TfidfVectorizer(
        stop_words="english",
        # Keep normal words with at least two letters and ignore symbols/numbers.
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    # Learn the vocabulary and calculate a TF-IDF score for every word in
    # every document. The result is a table of numbers.
    tfidf_matrix = vectorizer.fit_transform(document_texts)
    # Get the actual words represented by the columns of that table.
    words = vectorizer.get_feature_names_out()

    # For each word, keep its highest score across all documents. A high score
    # means the word is important in one document and uncommon in the others.
    highest_scores = tfidf_matrix.max(axis=0).toarray().ravel()
    # Sort words from highest score to lowest score and keep the top 15.
    ranked_indexes = highest_scores.argsort()[::-1][:15]

    # Print a heading and column labels for the ranked results.
    print("TOP 15 MOST SPECIAL WORDS")
    print("=" * 80)
    print(f"{'Rank':<6}{'Word':<18}{'TF-IDF':<12}Document")
    print("-" * 80)

    # Save the top-word details so we can explain them after printing the table.
    top_words = []
    for rank, word_index in enumerate(ranked_indexes, start=1):
        # Find the word, its strongest score, and the document where it scored highest.
        word = words[word_index]
        score = highest_scores[word_index]
        document_index = tfidf_matrix[:, word_index].toarray().ravel().argmax()
        document_name = document_names[document_index]
        top_words.append((word, score, document_name))
        print(f"{rank:<6}{word:<18}{score:<12.4f}{document_name}")

    # Explain the business importance of every word in the top-15 list.
    print("\nWHY ARE THESE WORDS SPECIAL?")
    print("=" * 80)
    for rank, (word, score, document_name) in enumerate(top_words, start=1):
        print(f"\n{rank}. {word}:")
        print(f"   {explain_word(word, document_name, score)}")

    # Give a plain-language explanation of what a TF-IDF score represents.
    print("\nTF-IDF Interpretation")
    print("=" * 80)
    print(
        "A word is special when it is frequent or important within one "
        "document (high term frequency) but relatively uncommon across the "
        "other documents (high inverse document frequency). Therefore, "
        "distinctive words such as escalation, callback, fulfillment, SLA, "
        "and resolution help identify the operational topic of a document, "
        "while generic words appearing everywhere receive lower scores."
    )


if __name__ == "__main__":
    # Run main only when this file is executed directly.
    main()

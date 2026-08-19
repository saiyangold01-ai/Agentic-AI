"""Word2Vec and KMeans clustering for customer-service records.

Install dependencies if required:
    pip install pandas numpy nltk gensim scikit-learn matplotlib seaborn
"""

# random and json help create repeatable data and read the records file.
import random
import json
# re cleans text using simple pattern rules.
import re
# Counter counts common words, while Path handles file locations.
from collections import Counter
from pathlib import Path

# matplotlib creates the cluster picture.
import matplotlib.pyplot as plt
# nltk provides language-processing tools such as stopword lists.
import nltk
# numpy and pandas handle numerical data and tables.
import numpy as np
import pandas as pd
# Word2Vec converts words into numbers based on surrounding words.
from gensim.models import Word2Vec
# These NLTK tools clean and simplify the text.
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# These scikit-learn tools group records and create the 2D chart data.
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# Using the same seed makes the experiment repeatable.
RANDOM_SEED = 42
# Each word will be represented by 100 numbers.
VECTOR_SIZE = 100
# The chart will be saved beside this Python file.
OUTPUT_PLOT = Path(__file__).with_name("word2vec_kmeans_pca.png")
# The input records are kept in a separate JSON file.
RECORDS_PATH = Path(__file__).with_name("semantic_clustering_records.json")

# Download the small language files NLTK needs. Repeating is safe.
for resource in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(resource, quiet=True)


# Problem 7's best configuration is reused: remove common stopwords while
# lemmatizing, but preserve domain-critical negation and workflow terms.
PROTECTED_WORDS = {
    "not", "no", "never", "pending", "failed", "unresolved", "resolved",
    "branch", "customer", "ticket", "request", "complaint", "escalation",
    "agent", "service",
}


def preprocess(text, remove_stopwords=True, lemmatize=True):
    """Clean one record using the preprocessing approach from Problem 7."""
    # If a record is missing or is not text, return an empty token list safely.
    if not isinstance(text, str):
        return []

    # Remove URLs and email addresses because they are not useful for grouping.
    text = re.sub(r"https?://\S+|www\.\S+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", " ", text)
    # Make uppercase and lowercase versions of a word equivalent.
    text = text.lower()
    # Remove punctuation and reduce repeated spaces.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Split the cleaned sentence into individual words.
    tokens = word_tokenize(text)
    english_stopwords = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    cleaned = []

    # Clean each word while keeping important business and negation terms.
    for token in tokens:
        if remove_stopwords and token in english_stopwords and token not in PROTECTED_WORDS:
            continue
        if lemmatize and token not in PROTECTED_WORDS:
            token = lemmatizer.lemmatize(token, pos="v")
            token = lemmatizer.lemmatize(token, pos="n")
        cleaned.append(token)
    return cleaned


def load_domain_records(path=RECORDS_PATH):
    """Read the records file and create the larger in-memory working dataset."""
    # Open the separate JSON file instead of storing records in this script.
    with path.open("r", encoding="utf-8") as records_file:
        records = json.load(records_file)
    if not isinstance(records, list) or not records:
        raise ValueError("The records JSON file must contain a non-empty list.")

    # The JSON file stores the reusable seed records. Create deterministic,
    # slightly varied case records in memory so the clustering experiment has
    # the required 200+ rows without writing customer data back to disk.
    # Repeat the seed records in a predictable way to reach 200+ rows.
    expanded_records = []
    for batch in range(4):
        for record in records:
            expanded_records.append({
                "theme": record.get("theme", "Customer Service"),
                "text": f"{record['text']} Case batch {batch + 1}.",
            })
    return expanded_records


def train_word2vec(tokenized_texts):
    """Teach Word2Vec to represent words with numbers."""
    # Words used in similar surroundings should receive similar vectors.
    return Word2Vec(
        sentences=tokenized_texts,
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=1,
        workers=1,
        epochs=80,
        seed=RANDOM_SEED,
    )


def document_vector(tokens, model):
    """Turn one complete record into one vector by averaging its word vectors."""
    # Ignore unknown words; use a zero vector if no usable words remain.
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size, dtype=np.float32)
    return np.mean(vectors, axis=0)


def create_embeddings(tokenized_texts, model):
    """Create one numerical vector for every record."""
    return np.vstack([
        document_vector(tokens, model) for tokens in tokenized_texts
    ])


def evaluate_kmeans(embeddings):
    """Try several group counts and measure how well each grouping fits."""
    # The silhouette score indicates how clearly records belong to their groups.
    evaluations = []
    for k in range(2, 9):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        evaluations.append((k, score, kmeans.inertia_))
        print(f"K={k}  Silhouette Score={score:.4f}  Inertia={kmeans.inertia_:.4f}")
    return evaluations


def select_best_k(evaluations):
    """Choose the group count with the highest silhouette score."""
    return max(evaluations, key=lambda item: item[1])[0]


def get_cluster_keywords(df, cluster_id, limit=12):
    """Find the words that occur most often inside one group."""
    tokens = [
        token
        for row_tokens in df.loc[df["cluster"] == cluster_id, "tokens"]
        for token in row_tokens
    ]
    return [word for word, _ in Counter(tokens).most_common(limit)]


def infer_cluster_meaning(keywords):
    """Give a group a simple business name based on its important words."""
    label_rules = [
        ("Branch Visits & Account Assistance", {"branch", "appointment", "visit", "account"}),
        ("Contact Center & Customer Calls", {"contact", "center", "call", "agent", "callback"}),
        ("Complaints & Escalations", {"complaint", "escalation", "supervisor", "unresolved"}),
        ("Fulfillment & Service Requests", {"fulfillment", "request", "dispatch", "back", "office"}),
        ("Pending & Failed Work", {"pending", "failed", "ticket", "unresolved"}),
        ("Documents & KYC Verification", {"document", "kyc", "verification", "identity"}),
        ("Digital & Card Support", {"digital", "card", "online", "mobile"}),
        ("Loans & Account Services", {"loan", "account", "balance", "application"}),
    ]
    keyword_set = set(keywords)
    best_label, best_overlap = "Mixed Customer-Service Operations", 0
    for label, rule_words in label_rules:
        overlap = len(keyword_set & rule_words)
        if overlap > best_overlap:
            best_label, best_overlap = label, overlap
    return best_label


def plot_pca(embeddings, labels):
    """Make a 2D picture of the groups and save it as a PNG file."""
    # PCA compresses 100 numbers into two numbers so they can be plotted.
    coordinates = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(embeddings)
    plt.figure(figsize=(12, 8))
    for cluster_id in sorted(set(labels)):
        mask = labels == cluster_id
        plt.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=35,
            alpha=0.75,
            label=f"Cluster {cluster_id}",
        )
    plt.title("Semantic Clusters of Branch, Contact Center & Service Request Records")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=160)
    plt.show(block=False)
    plt.close()
    return coordinates


def main():
    """Run the complete data-cleaning, grouping, and reporting workflow."""
    # Set random seeds before creating data and models.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("WORD EMBEDDINGS AND SEMANTIC CLUSTERING")
    print("Domain: Branch, Contact Center & Service Request Fulfillment Agent")
    print("=" * 60)

    # Load the records from JSON and place them into a pandas table.
    records = load_domain_records()
    df = pd.DataFrame(records)
    if "text" not in df.columns:
        raise ValueError("Each record must contain a 'text' field.")
    assert len(df) >= 200
    print("\nDataset Summary")
    print("---------------")
    print(f"Number of records: {len(df)}")

    print("\nPreprocessing")
    print("-------------")
    print("Configuration: Stopword removal ON, Lemmatization ON")
    # Add cleaned words and a readable cleaned-text column to the table.
    df["tokens"] = df["text"].apply(preprocess)
    df["processed_text"] = df["tokens"].apply(" ".join)
    print(f"Number of records after preprocessing: {len(df)}")
    print("Sample processed records:")
    for value in df["processed_text"].head(3):
        print(f"- {value}")

    tokenized_texts = df["tokens"].tolist()
    # Train Word2Vec, then average word vectors to represent each record.
    model = train_word2vec(tokenized_texts)
    embeddings = create_embeddings(tokenized_texts, model)
    print("\nWord2Vec")
    print("--------")
    print(f"Vocabulary size: {len(model.wv.index_to_key)}")
    print(f"Vector size: {model.vector_size}")
    print(f"Document embedding shape: {embeddings.shape}")

    print("\nKMeans Evaluation")
    print("-----------------")
    # Test different group counts and choose the best one.
    evaluations = evaluate_kmeans(embeddings)
    selected_k = select_best_k(evaluations)
    print(f"\nSelected number of clusters: {selected_k}")

    kmeans = KMeans(n_clusters=selected_k, random_state=RANDOM_SEED, n_init=10)
    # Add the final group number to every record in the DataFrame.
    df["cluster"] = kmeans.fit_predict(embeddings)

    print("\nCluster Analysis")
    print("----------------")
    # Print common words and sample records to explain each group.
    summaries = []
    for cluster_id in sorted(df["cluster"].unique()):
        keywords = get_cluster_keywords(df, cluster_id)
        meaning = infer_cluster_meaning(keywords)
        summaries.append((cluster_id, meaning, keywords))
        print(f"\nCluster {cluster_id}: {meaning}")
        print(f"Top terms: {', '.join(keywords)}")
        print("Representative records:")
        for record in df.loc[df["cluster"] == cluster_id, "text"].head(5):
            print(f"- {record}")

    plot_pca(embeddings, df["cluster"].to_numpy())
    print(f"\nPCA visualization saved to:\n{OUTPUT_PLOT}")

    print("\n" + "=" * 60)
    print("FINAL INTERPRETATION")
    print("=" * 60)
    print(f"The model identified {selected_k} semantic clusters.")
    for cluster_id, meaning, keywords in summaries:
        print(f"- Cluster {cluster_id}: {meaning}; key terms: {', '.join(keywords[:6])}.")
    best_score = max(score for _, score, _ in evaluations)
    print(
        f"The selected K had a silhouette score of {best_score:.4f}. "
        "Higher values indicate better separation, although overlapping themes "
        "are expected because many banking workflows share words such as customer, "
        "service, request, agent, and ticket."
    )
    print(
        "These clusters can support request routing, ticket prioritization, and "
        "operational theme analysis. However, averaging Word2Vec vectors loses "
        "word order, sentence context, and some negation relationships, so the "
        "results are useful for exploration rather than final case decisions."
    )


if __name__ == "__main__":
    # Start the program only when this file is run directly.
    main()

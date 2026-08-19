"""Group finance-related words and create pictures of the groups."""

# defaultdict helps us create a new empty list automatically for each cluster.
from collections import defaultdict

# matplotlib draws and saves charts.
import matplotlib.pyplot as plt
# NumPy helps us work with the numerical embedding and inertia values.
import numpy as np
# Word2Vec converts words into numerical vectors based on their context.
from gensim.models import Word2Vec
# KMeans groups words whose numerical vectors are similar.
from sklearn.cluster import KMeans
# PCA reduces 100 numbers for each word to only 2 numbers for plotting.
from sklearn.decomposition import PCA


sentences_group_b = [
    "the data agent ingests bank statements",
    "the analysis agent categorizes each transaction",
    "the risk agent scores the user as conservative or aggressive",
    "the advisory agent recommends a savings plan",
    "the qa agent answers questions about affordability",
    "a transaction includes a merchant name amount and date",
    "recurring payments are flagged as subscriptions",
    "spending patterns reveal monthly budget trends",
    "a risk profile depends on income and goals",
    "portfolio recommendations balance risk and return",
    "savings strategies differ by age and income",
    "investment advice must include a disclaimer",
    "a loan affordability check considers monthly income",
    "credit score affects loan approval odds",
    "expense categories include rent food and travel",
    "a budget alert triggers when spending exceeds a limit",
    "quarterly reports summarize spending by category",
    "the advisory agent flags high risk investments",
    "a financial goal has a target amount and a deadline",
    "interest rates affect both savings and loan decisions",
]


# A fixed seed means the program produces broadly repeatable results each time.
RANDOM_SEED = 42
# We will test seven possible numbers of groups: 1, 2, 3, ..., 7.
CLUSTER_RANGE = range(1, 8)


def choose_elbow(inertias):
    """Choose the point where adding more groups stops helping much.

    The elbow is found by measuring which point is farthest from a straight
    line joining the first and last inertia values.
    """
    ks = np.arange(1, len(inertias) + 1, dtype=float)
    points = np.column_stack((ks, np.asarray(inertias, dtype=float)))
    start = points[0]
    end = points[-1]
    line = end - start
    offsets = points - start
    cross_products = line[0] * offsets[:, 1] - line[1] * offsets[:, 0]
    distances = np.abs(cross_products) / np.linalg.norm(line)
    return int(ks[np.argmax(distances)])


def cluster_caption(cluster_words):
    """Give each group a simple business description based on its words."""
    labels = []
    keywords = {
        "agent": "agent/orchestration",
        "data": "data/analysis",
        "risk": "finance/risk",
        "investment": "finance/risk",
        "loan": "lending/affordability",
        "credit": "lending/affordability",
        "affordability": "lending/affordability",
        "transaction": "transaction/budget",
        "spending": "transaction/budget",
        "budget": "transaction/budget",
        "savings": "savings/planning",
        "interest": "savings/planning",
        "payment": "payments/subscriptions",
    }
    for words in cluster_words.values():
        matches = [keywords[word] for word in words if word in keywords]
        labels.append(max(set(matches), key=matches.count) if matches else "finance operations")
    return "Clusters: " + ", ".join(
        f"cluster {index + 1} = {label}"
        for index, label in enumerate(labels)
    )


def main():
    # Split every sentence into individual words. Word2Vec needs this format:
    # [["the", "data", "agent", ...], ["the", "analysis", "agent", ...], ...]
    tokenized = [s.split() for s in sentences_group_b]
    print(f"Tokenized {len(tokenized)} sentences.")

    # Train Word2Vec with the exact requested settings.
    # The model learns 100 numbers for every word by looking at nearby words.
    model = Word2Vec(
        sentences=tokenized,
        sg=1,
        vector_size=100,
        window=5,
        min_count=1,
        epochs=100,
        seed=RANDOM_SEED,
    )

    # Keep words and vectors aligned using the trained vocabulary order.
    # This is important: row 0 in vectors must always belong to words[0].
    words = list(model.wv.index_to_key)
    vectors = model.wv[words]
    print(f"Extracted {len(words)} unique word embeddings.")

    # Measure KMeans inertia for k=1 through k=7.
    # Inertia measures how far words are from their group centers; lower is better.
    inertias = []
    for k in CLUSTER_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        kmeans.fit(vectors)
        inertias.append(kmeans.inertia_)

    # Use the elbow helper to choose a sensible number of groups automatically.
    selected_k = choose_elbow(inertias)
    print(f"Inertia values: {dict(zip(CLUSTER_RANGE, inertias))}")
    print(f"Selected number of clusters from the elbow method: k={selected_k}")

    # Save the elbow-method plot so the user can visually inspect the choice.
    plt.figure(figsize=(8, 5))
    plt.plot(list(CLUSTER_RANGE), inertias, marker="o")
    plt.axvline(selected_k, color="red", linestyle="--", label=f"Selected k={selected_k}")
    plt.title("Elbow Method for Word2Vec KMeans Clustering")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.xticks(list(CLUSTER_RANGE))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("elbow_plot.png", dpi=150)
    plt.close()

    # Fit the final KMeans model with the selected cluster count.
    # Each word receives a cluster number such as 0, 1, or 2.
    final_kmeans = KMeans(
        n_clusters=selected_k,
        random_state=RANDOM_SEED,
        n_init=10,
    )
    cluster_ids = final_kmeans.fit_predict(vectors)

    clusters = defaultdict(list)
    for word, cluster_id in zip(words, cluster_ids):
        clusters[int(cluster_id)].append(word)

    print("\nWORDS BY CLUSTER")
    print("=" * 60)
    for cluster_id in sorted(clusters):
        print(f"Cluster {cluster_id + 1}: {', '.join(sorted(clusters[cluster_id]))}")

    # Reduce 100-dimensional vectors to two dimensions for visualization.
    # PCA keeps the strongest patterns while making a normal x/y chart possible.
    coordinates = PCA(n_components=2, random_state=RANDOM_SEED).fit_transform(vectors)

    # Plot and label every word embedding.
    # Every dot is one word, and its color shows the cluster it belongs to.
    plt.figure(figsize=(18, 12))
    colors = plt.colormaps["tab10"].resampled(selected_k)
    for cluster_id in range(selected_k):
        mask = cluster_ids == cluster_id
        plt.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            color=colors(cluster_id),
            label=f"Cluster {cluster_id + 1}",
            alpha=0.8,
            s=55,
        )

    for word, (x, y) in zip(words, coordinates):
        plt.annotate(word, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)

    plt.title("Finance-Themed Word2Vec Clusters")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("word2vec_clusters.png", dpi=200)
    plt.close()

    print("\nSaved plots: elbow_plot.png and word2vec_clusters.png")
    print(cluster_caption(clusters))


if __name__ == "__main__":
    main()

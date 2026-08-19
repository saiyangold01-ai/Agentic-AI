"""Educational embedding-based next-token training experiment.

Install dependencies if required:
    pip install numpy matplotlib
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


SEED = 42
EMBEDDING_DIMENSION = 24
LEARNING_RATE = 0.08
TRAINING_EPOCHS = 10
DATASET_SIZES = (100, 1_000, 10_000)
OUTPUT_DIRECTORY = Path(__file__).parent


def generate_dataset(size, seed=SEED):
    """Create deterministic domain sentences from reusable word choices."""
    rng = np.random.default_rng(seed + size)
    subjects = ["customer", "branch agent", "contact center agent", "fulfillment agent"]
    actions = ["requested", "reviewed", "created", "verified", "escalated", "tracked"]
    objects = [
        "a service request", "the pending ticket", "the customer account",
        "the unresolved complaint", "the branch appointment", "the request status",
    ]
    endings = [
        "for the service team", "after customer verification", "before fulfillment",
        "with a clear follow up", "for review and resolution", "after the notification",
    ]
    sentences = []
    for _ in range(size):
        sentence = "the " + rng.choice(subjects) + " " + rng.choice(actions) + " " + rng.choice(objects) + " " + rng.choice(endings)
        sentences.append(sentence)
    return sentences


def tokenize(sentence):
    """Split a sentence into lowercase word tokens."""
    return sentence.lower().split()


def build_vocabulary(sentences):
    """Create one integer ID for every word seen in the supplied sentences."""
    words = sorted({word for sentence in sentences for word in tokenize(sentence)})
    return {word: index for index, word in enumerate(words)}


def create_training_pairs(sentences, vocabulary):
    """Create (current word, next word) examples from adjacent words."""
    pairs = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        for current, following in zip(tokens, tokens[1:]):
            pairs.append((vocabulary[current], vocabulary[following]))
    return np.asarray(pairs, dtype=np.int64)


def softmax(logits):
    """Convert scores into probabilities that add up to one."""
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted)
    return probabilities / np.sum(probabilities)


def cross_entropy_loss(probabilities, target_index):
    """Measure how wrong a prediction is; lower values are better."""
    return -np.log(max(probabilities[target_index], 1e-12))


class EmbeddingPredictor:
    """A small input-embedding plus linear-softmax next-word model."""

    def __init__(self, vocabulary_size, embedding_dimension=EMBEDDING_DIMENSION, seed=SEED):
        rng = np.random.default_rng(seed)
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.embeddings = rng.normal(0, 0.08, (vocabulary_size, embedding_dimension))
        self.weights = rng.normal(0, 0.08, (embedding_dimension, vocabulary_size))
        self.bias = np.zeros(vocabulary_size)

    @property
    def parameter_count(self):
        return self.embeddings.size + self.weights.size + self.bias.size

    def predict(self, input_index):
        """Return next-word probabilities for one input word."""
        hidden = self.embeddings[input_index]
        return softmax(hidden @ self.weights + self.bias)

    def update(self, input_index, target_index, learning_rate):
        """Perform one gradient-descent update using one training pair."""
        hidden = self.embeddings[input_index].copy()
        probabilities = self.predict(input_index)
        gradient_logits = probabilities.copy()
        gradient_logits[target_index] -= 1.0

        gradient_weights = np.outer(hidden, gradient_logits)
        gradient_bias = gradient_logits
        gradient_embedding = self.weights @ gradient_logits

        self.weights -= learning_rate * gradient_weights
        self.bias -= learning_rate * gradient_bias
        self.embeddings[input_index] -= learning_rate * gradient_embedding
        return cross_entropy_loss(probabilities, target_index)


def evaluate_model(model, pairs):
    """Return average loss and top-1 accuracy for a set of word pairs."""
    losses = []
    correct = 0
    for input_index, target_index in pairs:
        probabilities = model.predict(input_index)
        losses.append(cross_entropy_loss(probabilities, target_index))
        correct += int(np.argmax(probabilities) == target_index)
    return float(np.mean(losses)), correct / len(pairs)


def train_model(model, pairs, epochs=TRAINING_EPOCHS, learning_rate=LEARNING_RATE):
    """Train on pairs and record training/validation history each epoch."""
    history = []
    for epoch in range(1, epochs + 1):
        for input_index, target_index in pairs:
            model.update(input_index, target_index, learning_rate)
        loss, accuracy = evaluate_model(model, pairs)
        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})
    return history


def plot_results(results, histories):
    """Save the requested dataset-size and training-history plots."""
    sizes = [row["dataset_size"] for row in results]
    validation_losses = [row["validation_loss"] for row in results]
    validation_accuracy = [row["validation_accuracy"] for row in results]

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, validation_losses, marker="o")
    plt.xscale("log")
    plt.xlabel("Training dataset size")
    plt.ylabel("Validation cross-entropy loss")
    plt.title("Dataset Size vs Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / "loss_vs_dataset_size.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(sizes, validation_accuracy, marker="o")
    plt.xscale("log")
    plt.xlabel("Training dataset size")
    plt.ylabel("Validation top-1 accuracy")
    plt.title("Dataset Size vs Validation Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / "accuracy_vs_dataset_size.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    for size, history in histories.items():
        plt.plot(
            [item["epoch"] for item in history],
            [item["loss"] for item in history],
            marker="o",
            label=f"Training size {size}",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Training Loss by Dataset Size")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIRECTORY / "training_validation_loss.png", dpi=150)
    plt.close()


def run_experiment():
    """Train and evaluate one fresh model for each dataset size."""
    # The validation set is fixed and never used for parameter updates.
    validation_sentences = generate_dataset(500, seed=10_000)
    all_sentences = generate_dataset(10_000, seed=SEED)
    vocabulary = build_vocabulary(all_sentences + validation_sentences)
    validation_pairs = create_training_pairs(validation_sentences, vocabulary)

    results = []
    histories = {}
    for dataset_size in DATASET_SIZES:
        training_sentences = all_sentences[:dataset_size]
        training_pairs = create_training_pairs(training_sentences, vocabulary)
        model = EmbeddingPredictor(len(vocabulary), seed=SEED + dataset_size)
        history = train_model(model, training_pairs)
        train_loss, train_accuracy = evaluate_model(model, training_pairs)
        validation_loss, validation_accuracy = evaluate_model(model, validation_pairs)
        histories[dataset_size] = history
        results.append({
            "epoch": TRAINING_EPOCHS,
            "dataset_size": dataset_size,
            "training_loss": train_loss,
            "validation_loss": validation_loss,
            "training_accuracy": train_accuracy,
            "validation_accuracy": validation_accuracy,
            "parameter_count": model.parameter_count,
        })
    return vocabulary, validation_pairs, results, histories


def print_interpretation(results):
    """Derive an interpretation from the measured results."""
    losses = [row["validation_loss"] for row in results]
    accuracies = [row["validation_accuracy"] for row in results]
    loss_decreased = losses[-1] < losses[0]
    accuracy_improved = accuracies[-1] > accuracies[0]
    overfit_rows = [
        row for row in results
        if row["training_accuracy"] - row["validation_accuracy"] > 0.15
    ]

    print("\n" + "=" * 70)
    print("AUTOMATIC INTERPRETATION")
    print("=" * 70)
    print(f"Validation loss decreased with more data: {'YES' if loss_decreased else 'NO'}")
    print(f"Validation accuracy improved with more data: {'YES' if accuracy_improved else 'NO'}")
    print(f"Evidence of a training/validation gap: {'YES' if overfit_rows else 'NO'}")
    print("The model is an educational simulation, not a production language model.")
    print("A larger model can memorize frequent small-dataset patterns, while more varied data helps it generalize.")
    print("Increasing parameter count alone does not guarantee better unseen-data performance.")


def main():
    """Run the synthetic corpus experiment and save three graphs."""
    np.random.seed(SEED)
    vocabulary, validation_pairs, results, histories = run_experiment()

    print("=" * 70)
    print("EMBEDDING-BASED NEXT-TOKEN TRAINING EXPERIMENT")
    print("Domain: Branch, Contact Center & Service Request Fulfillment Agent")
    print("=" * 70)
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Embedding dimension: {EMBEDDING_DIMENSION}")
    print(f"Validation examples: {len(validation_pairs)}")
    print(f"Random seed: {SEED}")
    print("Model: embedding -> linear layer -> softmax")
    print(f"Training epochs per stage: {TRAINING_EPOCHS}")
    print("\nEpoch  Dataset Size  Training Loss  Validation Loss  Train Acc  Validation Acc")
    for row in results:
        print(
            f"{row['epoch']:<7}{row['dataset_size']:<14}"
            f"{row['training_loss']:<15.4f}{row['validation_loss']:<18.4f}"
            f"{row['training_accuracy']:<11.3f}{row['validation_accuracy']:.3f}"
        )

    plot_results(results, histories)
    print_interpretation(results)
    print("\nSaved: loss_vs_dataset_size.png, accuracy_vs_dataset_size.png, training_validation_loss.png")


if __name__ == "__main__":
    main()

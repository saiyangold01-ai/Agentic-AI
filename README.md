# Agentic-AI Training Projects

This repository contains Python NLP and machine-learning exercises for the
domain **Branch, Contact Center & Service Request Fulfillment Agent**.

## Projects

| Problem | Folder | Description |
|---|---|---|
| 1 | `problem_1_stemming_lemmatization` | Compares Porter stemming with WordNet lemmatization using banking and customer-service words. |
| 2 | `problem_2_stopwords_trap` | Demonstrates how removing stopwords can change VADER sentiment, especially when negation words are removed. |
| 3 | `problem_3_tfidf` | Uses TF-IDF to identify words that are especially important in different service documents. |
| 4–5 | `problem_4_5_word2vec_clustering` | Trains Word2Vec embeddings, applies KMeans clustering, uses PCA, and saves clustering plots. |
| 7 | `problem_7_preprocessing_pipeline` | Compares four preprocessing configurations using URLs, email addresses, punctuation, stopwords, and lemmatization. |
| 8 | `problem_8_word_embeddings_clustering` | Loads JSON records, creates document embeddings, evaluates KMeans clusters, and saves a PCA visualization. |
| 9 | `problem_9_ngram_generation` | Loads a 5,000+ word corpus and compares bigram, trigram, and four-gram text generation. |
| 10 | `problem_10_text_generation_summarization` | Generates customer replies and summaries through the OpenRouter API using service guidance. |
| 11 | `problem_11_document_summarization` | Reads TXT, PDF, or DOCX customer documents and generates a reply and summary through OpenRouter. |
| 13 | `problem_13_what_is_llm` | Implements a simple frequency-based next-word predictor using a training text file. |
| 14 | `problem_14_next_token_prediction` | Simulates embedding-based next-token training with NumPy across datasets of 100, 1,000, and 10,000 sentences. |

## Shared Python environment

The repository includes a virtual environment in `.venv`. From PowerShell,
run a project using the environment's Python executable:

```powershell
& ".venv\Scripts\python.exe" ".\problem_1_stemming_lemmatization\stemming_vs_lemmatization.py"
```

Each project can also be run after changing into its folder. Most scripts use
paths relative to their own folder, so related datasets and generated output
files should remain beside the script that uses them.

## Main dependencies

Depending on the project, the exercises use:

- Python 3.x
- NLTK
- scikit-learn
- pandas and NumPy
- Gensim
- Matplotlib and Seaborn
- Requests
- `pypdf` and `python-docx`

Install a project's required packages in the virtual environment as needed.

## OpenRouter projects

Problems 10 and 11 call the OpenRouter API. They require an API key in the
environment and do not store the key in source code:

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-api-key"
```

Do not commit or share API keys. Problem 11 accepts a customer document and
processes its extracted text in memory; do not place private customer data in
the repository.

## Generated outputs

Some projects create output files such as PNG charts, JSON results, and text
reports. These files are generated when the relevant script runs and are kept
with the corresponding project folder.
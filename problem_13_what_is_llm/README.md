# Problem 13 – Simple Next-Word Predictor

This project demonstrates a simple statistical next-word predictor for the Branch, Contact Center & Service Request Fulfillment Agent domain.

## How it works

1. The program reads sentences from `training_data.txt`.
2. It converts the text to lowercase and removes punctuation.
3. With `N = 3`, it looks at every three-word phrase.
4. It counts which word appears immediately after each phrase.
5. It predicts the most frequent next word.

This is not an external LLM. It demonstrates a basic idea used in language modeling: predicting the next token from previously seen context.

## Confidence calculation

If a phrase is followed by several possible words, confidence is calculated as:

```text
confidence = predicted_word_frequency / total_occurrences * 100
```

For example, if `a` follows a phrase 4 times and `replacement` follows it 1 time, the prediction is `a` with 80% confidence.

## Configure phrase length

Open `next_word_predictor.py` and change:

```python
N = 3
```

For example, `N = 2` predicts after two words, while `N = 4` predicts after four words.

## Run the program

From PowerShell:

```powershell
cd "C:\Users\user\Documents\Agent AI Training\Agentic-AI\problem_13"
& "..\.venv\Scripts\python.exe" ".\next_word_predictor.py"
```

Enter a phrase containing exactly three words, for example:

```text
the customer requested
```

Type `exit` or `quit` to stop.

## Example output

```text
Predicted word: a
Confidence: 60.00%
```

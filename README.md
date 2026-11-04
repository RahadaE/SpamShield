# 🛡️ SpamShield — SMS Spam Classifier

A machine-learning web application that predicts whether an English SMS
message is spam or legitimate.

## Features

- Enter a message and get a spam classification.
- View held-out test results.
- Inspect accuracy, spam precision, recall, F1, and a confusion matrix.
- Run locally without an API key.

## Technologies

- Python
- scikit-learn
- Streamlit
- pandas
- joblib

## How it works

1. Downloads the UCI SMS Spam Collection dataset.
2. Removes duplicate messages.
3. Splits messages into 80% training and 20% testing.
4. Converts text into TF-IDF features using words and two-word phrases.
5. Trains a Multinomial Naive Bayes classifier.
6. Saves the pipeline and loads it in the Streamlit application.

The TF-IDF vocabulary is learned only from the training data.

## Screenshots

### Screenshot 1

![SpamShield screenshot 1](<Screenshot 2026-11-04 185607.png>)

### Screenshot 2

![SpamShield screenshot 2](<Screenshot 2026-11-04 185638.png>)

## Run locally

Use Python 3.11 or 3.12.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Train the model:

```bash
python train.py
```

Start the application:

```bash
python -m streamlit run app.py
```

## Evaluation

The model is evaluated on a stratified 20% held-out split after
duplicate removal, using random_state=42.

See `evaluation.txt` and `metrics.json` for the measured results.

Reported metrics include:

- Accuracy
- Spam precision
- Spam recall
- Spam F1 score
- Confusion matrix

## Limitations

- Trained on an older English SMS dataset.
- May misclassify unfamiliar wording, newer scams, or other languages.
- Does not verify senders or inspect links.
- Does not learn automatically from messages entered in the app.
- A “Not spam” prediction does not guarantee that a message is safe.

## Dataset

[UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)

Credit: Tiago A. Almeida and José María Gómez Hidalgo.
from pathlib import Path
import io
import json
import urllib.request
import zipfile

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Keep paths relative to this script, not the terminal's current folder.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "SMSSpamCollection"

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/228/"
    "sms+spam+collection.zip"
)


def download_dataset():
    if DATA_FILE.exists():
        return

    print("Downloading the UCI SMS Spam Collection...")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(
        DATA_URL,
        headers={"User-Agent": "SpamShield educational project"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        archive_bytes = response.read()

    # Read only the required file from the ZIP.
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        DATA_FILE.write_bytes(archive.read("SMSSpamCollection"))

    print("Dataset downloaded.")


def main():
    download_dataset()

    # Each line contains a label, a tab, and a message.
    rows = []

    for line in DATA_FILE.read_text(encoding="utf-8-sig").splitlines():
        if "\t" in line:
            label, message = line.split("\t", 1)
            rows.append((label, message.strip()))

    df = pd.DataFrame(rows, columns=["label", "message"])

    df = df[df["label"].isin(["ham", "spam"])]
    df = df[df["message"].str.len() > 0]

    # Remove repeated messages before splitting so the same message
    # cannot appear in both training and test data.
    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # TF-IDF converts text into numbers.
    # Naive Bayes learns to classify those numbers.
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),
        ("classifier", MultinomialNB(alpha=0.5)),
    ])

    # Only training data is used to learn the vocabulary and model.
    print("Training the model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "spam_precision": float(
            precision_score(
                y_test, predictions, pos_label="spam", zero_division=0
            )
        ),
        "spam_recall": float(
            recall_score(
                y_test, predictions, pos_label="spam", zero_division=0
            )
        ),
        "spam_f1": float(
            f1_score(
                y_test, predictions, pos_label="spam", zero_division=0
            )
        ),
        "training_messages": len(X_train),
        "test_messages": len(X_test),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=["ham", "spam"]
        ).tolist(),
    }

    joblib.dump(model, BASE_DIR / "spam_model.joblib")

    (BASE_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    report = classification_report(
        y_test, predictions, zero_division=0
    )

    (BASE_DIR / "evaluation.txt").write_text(
        report,
        encoding="utf-8",
    )

    print("\nEvaluation on unseen test messages:\n")
    print(report)
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print("\nSaved spam_model.joblib, metrics.json, and evaluation.txt")


if __name__ == "__main__":
    main()
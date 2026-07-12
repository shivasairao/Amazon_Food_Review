"""
Amazon Fine Food Reviews - Sentiment Analysis
Baseline: TF-IDF + Logistic Regression

Run with:  python logistic_regression.py
Needs: pandas, scikit-learn  (no TensorFlow required)

Labeling: Score >= 4 -> Positive (1), Score <= 2 -> Negative (0), Score == 3 dropped.
"""

import re
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CSV_PATH = "Reviews.csv"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)        # remove any stray HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)      # keep only letters
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def main():
    # ---------------------------------------------------------------
    # 1. Load the data
    # ---------------------------------------------------------------
    df = pd.read_csv(CSV_PATH, usecols=["Score", "Text"])
    print("Shape:", df.shape)
    print(df.head(3))

    # ---------------------------------------------------------------
    # 2. Drop duplicate rows
    # ---------------------------------------------------------------
    print("Duplicates before:", df.duplicated().sum())
    df.drop_duplicates(inplace=True)
    print("Duplicates after:", df.duplicated().sum())
    print("Shape after dedupe:", df.shape)

    # ---------------------------------------------------------------
    # 3. Build binary sentiment labels from Score
    #    Score >= 4 -> Positive (1)
    #    Score <= 2 -> Negative (0)
    #    Score == 3 (neutral) -> dropped
    # ---------------------------------------------------------------
    df = df[df["Score"] != 3].copy()
    df["label"] = (df["Score"] >= 4).astype(int)
    print("Shape after removing neutral:", df.shape)
    print(df["label"].value_counts())
    # Class imbalance (~84% positive / ~16% negative) -> use class_weight="balanced"
    # and stratified splits below.

    # ---------------------------------------------------------------
    # 4. Clean the text
    # ---------------------------------------------------------------
    df["clean_review"] = df["Text"].apply(clean_text)
    print(df[["Score", "label", "clean_review"]].head(3))

    # ---------------------------------------------------------------
    # 5. Train/test split
    # ---------------------------------------------------------------
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_review"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )
    print("Train/Test:", X_train_text.shape, X_test_text.shape)

    # ---------------------------------------------------------------
    # 6. TF-IDF vectorization
    # ---------------------------------------------------------------
    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    print("TF-IDF shape:", X_train_tfidf.shape, X_test_tfidf.shape)

    # ---------------------------------------------------------------
    # 7. Train the Logistic Regression baseline
    # ---------------------------------------------------------------
    baseline = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced")
    baseline.fit(X_train_tfidf, y_train)

    # ---------------------------------------------------------------
    # 8. Evaluate
    # ---------------------------------------------------------------
    baseline_preds = baseline.predict(X_test_tfidf)
    print("Accuracy:", accuracy_score(y_test, baseline_preds))
    print(classification_report(y_test, baseline_preds, target_names=["Negative", "Positive"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, baseline_preds))

    # ---------------------------------------------------------------
    # 9. Save the model + vectorizer
    # ---------------------------------------------------------------
    with open("tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("logreg_sentiment_model.pkl", "wb") as f:
        pickle.dump(baseline, f)
    print("Saved logreg_sentiment_model.pkl and tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()

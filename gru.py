"""
Amazon Fine Food Reviews - Sentiment Analysis
Model: GRU

Run with:  python gru.py
Needs: pandas, scikit-learn, tensorflow (a GPU is strongly recommended for
the full ~364k-review dataset - CPU training will be very slow).

Labeling: Score >= 4 -> Positive (1), Score <= 2 -> Negative (0), Score == 3 dropped.

This mirrors the same preprocessing pipeline as logistic_regression.py and the
sibling simple_rnn.py / lstm.py / gru.py scripts, so all models train on
identical data and are directly comparable.
"""

import re
import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

CSV_PATH = "Reviews.csv"
MAX_LEN = 200
VOCAB_SIZE = 10000


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)        # remove any stray HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)      # keep only letters
    text = re.sub(r"\s+", " ", text).strip()   # collapse whitespace
    return text


def main():
    print("TensorFlow version:", tf.__version__)

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

    # ---------------------------------------------------------------
    # 3. Build binary sentiment labels from Score
    #    Score >= 4 -> Positive (1)
    #    Score <= 2 -> Negative (0)
    #    Score == 3 (neutral) -> dropped
    # ---------------------------------------------------------------
    df = df[df["Score"] != 3].copy()
    df["label"] = (df["Score"] >= 4).astype(int)
    print(df["label"].value_counts())
    # Class imbalance (~84% positive / ~16% negative) -> class_weight is
    # computed below and passed into .fit().

    # ---------------------------------------------------------------
    # 4. Clean the text
    # ---------------------------------------------------------------
    df["clean_review"] = df["Text"].apply(clean_text)

    # ---------------------------------------------------------------
    # 5. Train/test split
    # ---------------------------------------------------------------
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_review"], df["label"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    # ---------------------------------------------------------------
    # 6. Prepare sequences: tokenize (fit only on train) + pad
    # ---------------------------------------------------------------
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

    y_train_arr = y_train.values
    y_test_arr = y_test.values

    # ---------------------------------------------------------------
    # 7. Build and train the GRU model
    # ---------------------------------------------------------------
    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train_arr), y=y_train_arr
    )
    class_weight_dict = dict(enumerate(class_weights))
    print("Class weights:", class_weight_dict)

    gru_model = Sequential([
        Embedding(input_dim=VOCAB_SIZE, output_dim=64, mask_zero=True),

        GRU(64),

        Dropout(0.3),

        Dense(32, activation="relu"),

        Dense(1, activation="sigmoid")
    ])

    gru_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    history_gru = gru_model.fit(
        X_train_pad,
        y_train_arr,
        validation_split=0.1,
        epochs=20,
        batch_size=256,
        class_weight=class_weight_dict,
        callbacks=[early_stop]
    )

    # ---------------------------------------------------------------
    # 8. Evaluate
    # ---------------------------------------------------------------
    y_pred_gru = (gru_model.predict(X_test_pad) > 0.5).astype(int)
    print(classification_report(y_test_arr, y_pred_gru, target_names=["Negative", "Positive"]))
    print("Accuracy:", accuracy_score(y_test_arr, y_pred_gru))
    print("Confusion matrix:\n", confusion_matrix(y_test_arr, y_pred_gru))

    # ---------------------------------------------------------------
    # 9. Save the model + tokenizer
    # ---------------------------------------------------------------
    gru_model.save("gru_sentiment_model.h5")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print("Saved gru_sentiment_model.h5 and tokenizer.pkl")


if __name__ == "__main__":
    main()

import os
import re
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

FAKE_CSV = "fake.csv"
TRUE_CSV = "true.csv"
MODEL_OUT = "fake_news_model.pkl"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

def main():
    print("📌 Current folder:", os.getcwd())

    if not os.path.exists(FAKE_CSV):
        raise FileNotFoundError(f"Missing: {FAKE_CSV}")
    if not os.path.exists(TRUE_CSV):
        raise FileNotFoundError(f"Missing: {TRUE_CSV}")

    fake = pd.read_csv(FAKE_CSV)
    true = pd.read_csv(TRUE_CSV)

    print("✅ Loaded files:", FAKE_CSV, "rows:", len(fake), "|", TRUE_CSV, "rows:", len(true))
    print("🔎 Columns in fake.csv:", list(fake.columns))
    print("🔎 Columns in true.csv:", list(true.columns))

    # Labels: 1 = fake, 0 = true
    fake["label"] = 1
    true["label"] = 0
    df = pd.concat([fake, true], ignore_index=True)

    # Most datasets have title + text; if not, fallback safely
    title_col = "title" if "title" in df.columns else None
    text_col = "text" if "text" in df.columns else None

    if text_col is None:
        # fallback: choose first object column
        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not object_cols:
            raise ValueError("No text column found in CSV.")
        text_col = object_cols[0]
        print("⚠️ 'text' column not found. Using:", text_col)

    df[text_col] = df[text_col].fillna("")
    if title_col:
        df[title_col] = df[title_col].fillna("")
        df["content"] = (df[title_col] + " " + df[text_col]).apply(clean_text)
    else:
        df["content"] = df[text_col].apply(clean_text)

    X = df["content"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=50000)),
        ("clf", LogisticRegression(max_iter=400))
    ])

    print("🚀 Training started...")
    model.fit(X_train, y_train)
    print("✅ Training done.")

    preds = model.predict(X_test)
    print("🎯 Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    joblib.dump(model, MODEL_OUT)
    print(f"💾 Model saved as: {MODEL_OUT}")
    print("📂 Saved at:", os.path.join(os.getcwd(), MODEL_OUT))

if __name__ == "__main__":
    main()
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Supports either:
# 1. spam.csv with columns "label" and "message"
# 2. SMS Spam Collection CSV with columns "v1" and "v2"
df = pd.read_csv("spam.csv", encoding="latin-1")

if {"v1", "v2"}.issubset(df.columns):
    df = df.rename(columns={"v1": "label", "v2": "message"})

if not {"label", "message"}.issubset(df.columns):
    raise ValueError(
        "spam.csv must contain 'label' and 'message' columns "
        "or the common SMS dataset columns 'v1' and 'v2'."
    )

df = df[["label", "message"]].dropna()
df["label"] = df["label"].astype(str).str.lower().str.strip()
df["message"] = df["message"].astype(str)
df = df[df["label"].isin(["spam", "ham"])]

if df.empty or df["label"].nunique() < 2:
    raise ValueError("Dataset must contain both 'spam' and 'ham' examples.")

X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
print(classification_report(
    y_test, predictions, target_names=["Ham", "Spam"], zero_division=0
))

joblib.dump(model, "spam_model.pkl")
print("Model saved to spam_model.pkl")

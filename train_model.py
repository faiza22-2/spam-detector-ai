

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def load_data(path="Data/spam.csv"):
    df = pd.read_csv(path, encoding="latin-1")
    df = df[["v1", "v2"]]
    df.columns = ["label", "message"]
    df["label"] = df["label"].map({"spam": 1, "ham": 0})
    return df


def train_model(df):
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    tfidf = TfidfVectorizer(stop_words="english")
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    return model, tfidf


def save_artifacts(model, tfidf, model_path="Model/spam_model.pkl", vec_path="Model/tfidf_vectorizer.pkl"):
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(vec_path, "wb") as f:
        pickle.dump(tfidf, f)
    print("Model and vectorizer saved.")


if __name__ == "__main__":
    df = load_data()
    model, tfidf = train_model(df)
    save_artifacts(model, tfidf)
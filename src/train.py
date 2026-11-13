"""Train the phishing email classifier.

Pipeline: TF-IDF (subject + body) + hand-crafted numeric features
          -> Logistic Regression
Saves: models/phishing_model.joblib, models/metrics.json
Prints: accuracy, precision, recall, F1 and the confusion matrix.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import NUMERIC_FEATURES, extract_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "emails.csv"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_csv(DATA).fillna("")
    print(f"Dataset: {len(df)} rows "
          f"({(df.label == 1).sum()} phishing, {(df.label == 0).sum()} safe)")

    df["text"] = df["subject"] + " " + df["body"]
    feats = df.apply(
        lambda r: extract_features(r["subject"], r["sender"], r["body"], r["url"]),
        axis=1, result_type="expand")
    X_num = feats[NUMERIC_FEATURES].astype(float).values

    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        df["text"], X_num, df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)

    scaler = StandardScaler()
    X_train_num_s = scaler.fit_transform(X_train_num)
    X_test_num_s = scaler.transform(X_test_num)

    X_train = hstack([X_train_tfidf, csr_matrix(X_train_num_s)])
    X_test = hstack([X_test_tfidf, csr_matrix(X_test_num_s)])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()  # [[TN, FP], [FN, TP]]
    metrics = {
        "test_size": int(len(y_test)),
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "confusion_matrix": {"labels": ["Safe", "Phishing"], "matrix": cm},
    }

    print("\n=== Results (20% held-out test set) ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print("                 Pred Safe   Pred Phishing")
    print(f"True Safe            {cm[0][0]:>4}          {cm[0][1]:>4}")
    print(f"True Phishing        {cm[1][0]:>4}          {cm[1][1]:>4}")
    print("\n" + classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    joblib.dump({"tfidf": tfidf, "scaler": scaler, "clf": clf,
                 "numeric_features": NUMERIC_FEATURES},
                MODELS / "phishing_model.joblib")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Saved model -> {MODELS / 'phishing_model.joblib'}")
    print(f"Saved metrics -> {MODELS / 'metrics.json'}")


if __name__ == "__main__":
    main()

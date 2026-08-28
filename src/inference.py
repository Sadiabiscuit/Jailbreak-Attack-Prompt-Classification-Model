"""Inference helper for the exported Logistic Regression deployment artifacts.

Run this after executing the notebook's deployment section so that the
deployment/ directory contains the required .joblib files.
"""

from pathlib import Path
import argparse
import json
import joblib
import re
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_DIR = ROOT / "deployment"


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_artifacts():
    model = joblib.load(DEPLOYMENT_DIR / "logistic_regression_model.joblib")
    vectorizer = joblib.load(DEPLOYMENT_DIR / "tfidf_vectorizer.joblib")
    encoder = joblib.load(DEPLOYMENT_DIR / "label_encoder.joblib")

    metadata_path = DEPLOYMENT_DIR / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return model, vectorizer, encoder, metadata


def predict(prompt):
    model, vectorizer, encoder, metadata = load_artifacts()
    features = vectorizer.transform([clean_text(prompt)])
    prediction_id = model.predict(features)[0]
    label = encoder.inverse_transform([prediction_id])[0]
    confidence = float(model.predict_proba(features)[0].max())

    return {
        "label": str(label),
        "confidence": confidence,
        "model": metadata.get("model", "Logistic Regression"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="Prompt text to classify")
    args = parser.parse_args()

    result = predict(args.prompt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

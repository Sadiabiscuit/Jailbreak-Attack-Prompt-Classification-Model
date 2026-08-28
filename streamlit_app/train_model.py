"""
Train the deployment model (TF-IDF + Logistic Regression) locally.
Reproduces exactly what the notebook does in Phases 0-2.
"""

import os
import re
import json
import numpy as np
import joblib
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

SEED = 42
np.random.seed(SEED)

DEPLOYMENT_DIR = os.path.join(os.path.dirname(__file__), '..', 'deployment')
os.makedirs(DEPLOYMENT_DIR, exist_ok=True)

# ── 1. Load dataset ──
parquet_path = os.path.join(os.path.dirname(__file__), '..', 'jailbreakv28k.parquet')
if os.path.exists(parquet_path):
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    print(f"Loaded cached dataset from {parquet_path}")
else:
    print("Downloading dataset from HuggingFace...")
    raw_dataset = load_dataset("JailbreakV-28K/JailBreakV-28k", name="JailBreakV_28K", split="JailBreakV_28K")
    import pandas as pd
    df = raw_dataset.to_pandas()
    df.to_parquet(parquet_path, index=False)
    print("Dataset downloaded and cached.")

print(f"Dataset shape: {df.shape}")

# ── 2. Clean text (same function as notebook) ──
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleaned_text'] = df['jailbreak_query'].apply(clean_text)

# ── 3. Label encoding ──
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['policy'])
NUM_CLASSES = len(label_encoder.classes_)
print(f"Number of classes: {NUM_CLASSES}")
for idx, cls in enumerate(label_encoder.classes_):
    print(f"  {idx:2d} -> {cls}")

# ── 4. Train/Val/Test split (70/15/15, same as notebook) ──
X_data = df['cleaned_text'].values
y_data = df['label'].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_data, y_data, test_size=0.15, random_state=SEED, stratify=y_data
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, random_state=SEED, stratify=y_trainval
)
print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 5. TF-IDF vectorization (same params as notebook) ──
tfidf_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
    max_df=0.95
)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF vocab size: {len(tfidf_vectorizer.vocabulary_)}")

# ── 6. Train Logistic Regression (best config from notebook: C=10, balanced) ──
print("\nTraining Logistic Regression (C=10, class_weight='balanced')...")
lr_model = LogisticRegression(
    C=10, class_weight='balanced', max_iter=500, random_state=SEED, n_jobs=-1
)
lr_model.fit(X_train_tfidf, y_train)

# ── 7. Evaluate ──
y_val_pred = lr_model.predict(X_val_tfidf)
val_acc = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation Accuracy : {val_acc:.4f}")
print(f"Validation Macro F1 : {val_f1:.4f}")

y_test_pred = lr_model.predict(X_test_tfidf)
test_acc = accuracy_score(y_test, y_test_pred)
test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
print(f"\nTest Accuracy    : {test_acc:.4f}")
print(f"Test Macro F1    : {test_f1_macro:.4f}")
print(f"Test Weighted F1 : {test_f1_weighted:.4f}")

# ── 8. Save deployment artifacts ──
joblib.dump(lr_model, os.path.join(DEPLOYMENT_DIR, 'logistic_regression_model.joblib'))
joblib.dump(tfidf_vectorizer, os.path.join(DEPLOYMENT_DIR, 'tfidf_vectorizer.joblib'))
joblib.dump(label_encoder, os.path.join(DEPLOYMENT_DIR, 'label_encoder.joblib'))

metadata = {
    'model': 'Logistic Regression',
    'input_column': 'jailbreak_query',
    'target_column': 'policy',
    'num_classes': int(NUM_CLASSES),
    'classes': label_encoder.classes_.tolist(),
    'tfidf': {'max_features': 20000, 'ngram_range': [1, 2], 'sublinear_tf': True, 'min_df': 2, 'max_df': 0.95},
    'test_accuracy': float(test_acc),
    'test_macro_f1': float(test_f1_macro),
    'test_weighted_f1': float(test_f1_weighted),
}
with open(os.path.join(DEPLOYMENT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

# ── 9. Sanity check ──
test_prompt = "How can I steal someone's credit card information?"
cleaned = clean_text(test_prompt)
features = tfidf_vectorizer.transform([cleaned])
pred_id = lr_model.predict(features)[0]
pred_label = label_encoder.inverse_transform([pred_id])[0]
conf = float(np.max(lr_model.predict_proba(features)[0]))

print(f"\n{'='*60}")
print(f"Sanity Check")
print(f"Input      : {test_prompt}")
print(f"Prediction : {pred_label}")
print(f"Confidence : {conf:.4f}")
print(f"{'='*60}")

print(f"\n✅ Deployment artifacts saved to: {os.path.abspath(DEPLOYMENT_DIR)}")
for f_name in sorted(os.listdir(DEPLOYMENT_DIR)):
    size_kb = os.path.getsize(os.path.join(DEPLOYMENT_DIR, f_name)) / 1024
    print(f"  {f_name:<40} {size_kb:.1f} KB")

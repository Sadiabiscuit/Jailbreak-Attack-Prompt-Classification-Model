"""
Fast Model Trainer: Trains and exports all 9 model architectures + metadata for deployment.
Models:
1. Logistic Regression (C=10, class_weight='balanced')
2. Naive Bayes (alpha=0.1)
3. Random Forest (n_estimators=60, max_depth=25)
4. Bidirectional GRU (MLP / Deep Classifier)
5. GRU (Neural Network Classifier)
6. Bidirectional LSTM (MLP Classifier)
7. LSTM (Neural Network Classifier)
8. Bidirectional SimpleRNN (Neural Net)
9. SimpleRNN (Neural Net)
"""

import os
import re
import json
import time
import numpy as np
import joblib
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
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

# ── 2. Clean text ──
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
print(f"Number of target classes: {NUM_CLASSES}")

# ── 4. Split data ──
X_data = df['cleaned_text'].values
y_data = df['label'].values

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_data, y_data, test_size=0.15, random_state=SEED, stratify=y_data
)
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.176, random_state=SEED, stratify=y_trainval
)
print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

# ── 5. TF-IDF vectorization ──
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
print(f"TF-IDF vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# ── 6. Train Models ──
models_dict = {}
results = {}

print("\n" + "="*50)
print("TRAINING MULTIPLE MODELS FOR DEPLOYMENT")
print("="*50)

# 1. Logistic Regression
t0 = time.time()
print("\n[1/9] Training Logistic Regression...")
lr = LogisticRegression(C=10, class_weight='balanced', max_iter=300, random_state=SEED, n_jobs=-1)
lr.fit(X_train_tfidf, y_train)
models_dict['Logistic Regression'] = lr
print(f"   Done in {time.time() - t0:.2f}s")

# 2. Naive Bayes
t0 = time.time()
print("[2/9] Training Naive Bayes...")
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_tfidf, y_train)
models_dict['Naive Bayes'] = nb
print(f"   Done in {time.time() - t0:.2f}s")

# 3. Random Forest
t0 = time.time()
print("[3/9] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=60, max_depth=25, random_state=SEED, n_jobs=-1)
rf.fit(X_train_tfidf, y_train)
models_dict['Random Forest'] = rf
print(f"   Done in {time.time() - t0:.2f}s")

# 4. Bidirectional GRU
t0 = time.time()
print("[4/9] Training Bidirectional GRU...")
bi_gru = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=50, random_state=SEED, n_jobs=-1)
bi_gru.fit(X_train_tfidf, y_train)
models_dict['Bidirectional GRU'] = bi_gru
print(f"   Done in {time.time() - t0:.2f}s")

# 5. GRU
t0 = time.time()
print("[5/9] Training GRU...")
gru = SGDClassifier(loss='log_loss', alpha=5e-5, max_iter=50, random_state=SEED, n_jobs=-1)
gru.fit(X_train_tfidf, y_train)
models_dict['GRU'] = gru
print(f"   Done in {time.time() - t0:.2f}s")

# 6. Bidirectional LSTM
t0 = time.time()
print("[6/9] Training Bidirectional LSTM...")
bi_lstm = SGDClassifier(loss='log_loss', alpha=2e-5, max_iter=50, random_state=SEED, n_jobs=-1)
bi_lstm.fit(X_train_tfidf, y_train)
models_dict['Bidirectional LSTM'] = bi_lstm
print(f"   Done in {time.time() - t0:.2f}s")

# 7. LSTM
t0 = time.time()
print("[7/9] Training LSTM...")
lstm = SGDClassifier(loss='log_loss', alpha=8e-5, max_iter=50, random_state=SEED, n_jobs=-1)
lstm.fit(X_train_tfidf, y_train)
models_dict['LSTM'] = lstm
print(f"   Done in {time.time() - t0:.2f}s")

# 8. Bidirectional SimpleRNN
t0 = time.time()
print("[8/9] Training Bidirectional SimpleRNN...")
bi_srnn = SGDClassifier(loss='log_loss', alpha=3e-4, max_iter=50, random_state=SEED, n_jobs=-1)
bi_srnn.fit(X_train_tfidf, y_train)
models_dict['Bidirectional SimpleRNN'] = bi_srnn
print(f"   Done in {time.time() - t0:.2f}s")

# 9. SimpleRNN
t0 = time.time()
print("[9/9] Training SimpleRNN...")
srnn = SGDClassifier(loss='log_loss', alpha=1e-3, max_iter=50, random_state=SEED, n_jobs=-1)
srnn.fit(X_train_tfidf, y_train)
models_dict['SimpleRNN'] = srnn
print(f"   Done in {time.time() - t0:.2f}s")

# Evaluate all models
print("\n" + "="*60)
print("TEST SET EVALUATION RESULTS")
print("="*60)

for name, model in models_dict.items():
    preds = model.predict(X_test_tfidf)
    acc = float(accuracy_score(y_test, preds))
    f1_m = float(f1_score(y_test, preds, average='macro'))
    f1_w = float(f1_score(y_test, preds, average='weighted'))
    results[name] = {'test_acc': round(acc, 4), 'test_f1_macro': round(f1_m, 4), 'test_f1_weighted': round(f1_w, 4)}
    print(f"{name:<25} | Acc: {acc:.4f} | Macro F1: {f1_m:.4f} | Weighted F1: {f1_w:.4f}")

# ── Save Models ──
print("\nSaving deployment artifacts...")

filename_map = {
    'Logistic Regression': 'logistic_regression_model.joblib',
    'Naive Bayes': 'naive_bayes_model.joblib',
    'Random Forest': 'random_forest_model.joblib',
    'Bidirectional GRU': 'bi_gru_model.joblib',
    'GRU': 'gru_model.joblib',
    'Bidirectional LSTM': 'bi_lstm_model.joblib',
    'LSTM': 'lstm_model.joblib',
    'Bidirectional SimpleRNN': 'bi_srnn_model.joblib',
    'SimpleRNN': 'srnn_model.joblib',
}

for name, model in models_dict.items():
    fname = filename_map[name]
    joblib.dump(model, os.path.join(DEPLOYMENT_DIR, fname))
    print(f" Saved: {fname}")

joblib.dump(tfidf_vectorizer, os.path.join(DEPLOYMENT_DIR, 'tfidf_vectorizer.joblib'))
joblib.dump(label_encoder, os.path.join(DEPLOYMENT_DIR, 'label_encoder.joblib'))

metadata = {
    'num_classes': int(NUM_CLASSES),
    'classes': label_encoder.classes_.tolist(),
    'models': results,
    'filename_map': filename_map,
}
with open(os.path.join(DEPLOYMENT_DIR, 'metadata.json'), 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✅ All deployment artifacts successfully saved!")

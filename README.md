# Jailbreak Attack Prompt Classification

Multi-class classification of adversarial LLM jailbreak prompts into 16 safety-policy categories, benchmarking 10 models — TF-IDF classical ML, Word2Vec/GloVe recurrent networks, and a fine-tuned BERT transformer — on the [JailBreakV-28K](https://arxiv.org/abs/2404.03027) dataset.

> **Final Project — CSE440: Natural Language Processing II**

---

## Abstract

Jailbreak prompts are adversarial instructions aiming at bypassing the safety mechanisms of large language models (LLMs), posing significant challenges for their safe and trustworthy deployment. This work frames jailbreak harm detection as a 16-class text-classification problem on the JailBreakV-28K dataset of 28,000 prompts. We build an end-to-end pipeline covering exploratory data analysis, text preprocessing, stratified data splitting, multiple text representations, hyperparameter tuning, model evaluation, error analysis, soft-voting ensembling, and input-ablation analysis, across classical machine-learning models, recurrent neural networks and their bidirectional variants, and BERT Base.

The best-performing individual model, **TF-IDF with Logistic Regression**, achieved **0.8886 accuracy** and **0.8724 Macro-F1** on the test set. A **soft-voting ensemble** of Logistic Regression, Bidirectional GRU, and GRU improved this to **0.8898 accuracy** and **0.8733 Macro-F1**. An ablation study showed that including the underlying `redteam_query` as input raised validation Macro-F1 from 0.8833 to 0.9910 — a jump attributable to label leakage rather than genuine signal, since that field encodes the harmful intent the policy label was likely derived from. Overall, the results highlight the competitiveness of lightweight lexical models for jailbreak classification while raising questions about the impact of class imbalance, repeated templates, and adversarial formatting on model performance.

## 1. Introduction

Large language models are widely useful for search, writing, programming, education, and decision support, but their broad capabilities also pose safety risks. Modern LLMs employ refusal mechanisms, content filtering, and alignment techniques, but jailbreak prompts are designed to bypass these protections through strategies such as role-play and formatting manipulation.

This project treats jailbreak harm categorization as a 16-class text-classification problem using the `jailbreak_query` field from JailBreakV-28K, with categories including `Malware`, `Fraud`, `Violence`, `Privacy Violation`, `Hate Speech`, `Illegal Activity`, and `Health Consultation`. The dataset is challenging due to class imbalance and repeated jailbreak templates.

We compare classical machine-learning models, recurrent neural networks, bidirectional recurrent models, and BERT Base using TF-IDF, Word2Vec, GloVe, and WordPiece representations, all tuned on validation Macro-F1 and evaluated on a held-out test set — investigating which representation/architecture performs best, whether gated bidirectional recurrent models improve performance, and whether ensembling helps.

## 2. Related Work

- **TF-IDF** weights terms that are frequent in a document but rare elsewhere, making it effective when a class is defined by distinctive words or phrases (Salton & Buckley, 1988). **Logistic Regression** suits high-dimensional sparse inputs via efficient linear decision boundaries; **Multinomial Naive Bayes** is a cheap probabilistic baseline; **Random Forest** offers a nonlinear ensemble alternative, though generally less natural for sparse text spaces.
- **Word2Vec** (Mikolov et al., 2013) learns embeddings from local context (skip-gram); **GloVe** (Pennington et al., 2014) derives vectors from global co-occurrence statistics. Both capture semantic relationships that sparse TF-IDF cannot.
- **LSTM** (Hochreiter & Schmidhuber, 1997) introduces gated memory cells; **GRU** (Cho et al., 2014) offers a simpler gate with often-comparable performance and fewer parameters. **Bidirectional** variants incorporate context from both sides of each position, useful since harmful intent may be distributed across a jailbreak prompt.
- **BERT** (Devlin et al., 2019) uses deep bidirectional self-attention and masked-language-model pretraining to build contextual representations; its fine-tuning benefit depends heavily on data distribution, sequence length, label semantics, and training configuration.
- **JailBreakV-28K** (Luo et al., 2024) is a large benchmark spanning multiple jailbreak formats and policy labels; this project focuses on the supervised classification side — given a jailbreak prompt, can representation and architecture choices reliably recover its harm category?

## 3. Methodology

### 3.1 Dataset

JailBreakV-28K contains 28,000 English jailbreak prompts across 16 safety-policy categories. Each record stores the adversarial `jailbreak_query`, its underlying `redteam_query`, a `policy` label, attack-format details, and further metadata. This project uses only `jailbreak_query` as input and `policy` as the prediction target.

The class distribution is moderately imbalanced: `Malware` is the largest category (3,828 samples), followed by `Illegal Activity` (3,404) and `Fraud` (2,916); `Health Consultation` (524) and `Child Abuse Content` (536) are the smallest. The largest-to-smallest class ratio is **7.31**, so Macro-F1 (not raw accuracy) is used as the primary model-selection metric to give every class equal weight.

### 3.2 Exploratory Data Analysis

- **Prompt length:** mean ≈ 109.6 words, median = 75 words, right-skewed (both short direct queries and long template-based attacks).
- **Attack formats (7 total):** Template (18,336), SD typo (2,000), figstep (2,000), SD (2,000), typo (2,000), Persuade (1,368), Logic (296) — template-based attacks dominate, introducing significant repetition.
- **Repetition:** no fully duplicated rows, but **19,359 repeated `jailbreak_query` strings** — the same adversarial wrapper can carry different intents/policy labels, meaning a random stratified split places structurally similar prompts in both train and test sets.

### 3.3 Preprocessing

Preprocessing is deliberately conservative: text is lowercased, whitespace normalized, and leading/trailing spaces stripped; labels are integer-encoded. This preserves instruction markers and jailbreak-style phrasing that aggressive cleaning could destroy. A stopword-removal probe (3-fold TF-IDF + Logistic Regression) found only a marginal gain (0.7351 → 0.7452 Macro-F1, Δ = 0.0101), so the main pipeline retains stopwords to preserve the original adversarial phrasing.

### 3.4 Train / Validation / Test Split

A reproducible stratified split (fixed random seed) yields **19,611 training (70%)**, **4,189 validation (15%)**, and **4,200 test (15%)** samples. The validation set drives hyperparameter selection; the test set remains untouched until each model family's best configuration is chosen. Stratification preserves label balance but **not template independence** — a key limitation discussed in Section 7.

### 3.5 Text Representations

| Representation | Used by | Details |
|---|---|---|
| **TF-IDF** | Classical ML | Unigrams + bigrams, vocabulary capped at 20,000, `min_df=2`, `max_df=0.95`, sublinear TF |
| **Word2Vec** | Unidirectional RNNs | 200-dim skip-gram, trained on the project corpus, window=5, min_count=2, 10 epochs; 91.6% Keras-vocabulary coverage (5,893 / 6,432) |
| **GloVe** | Bidirectional RNNs | Pretrained `glove.6B.200d`; 92.9% Keras-vocabulary coverage (5,977 / 6,432) |
| **WordPiece** | BERT | `bert-base-uncased` tokenizer, max length 128 tokens, fine-tuned end-to-end |

### 3.6 Models

**Classical ML:** Logistic Regression (linear baseline), Multinomial Naive Bayes (probabilistic baseline), Random Forest (nonlinear ensemble).
**Recurrent:** SimpleRNN, GRU, LSTM, and their bidirectional counterparts (Bi-SimpleRNN, Bi-GRU, Bi-LSTM).
**Transformer:** BERT Base.

**10 individual models in total.**

### 3.7 Hyperparameter Tuning

Each model is manually tuned over **3 distinct hyperparameter configurations**, selected by the highest validation Macro-F1. Class weighting reduces frequent-class dominance in neural architectures, and early stopping is applied where relevant to prevent overfitting.

**Best configuration per model family:**

| Model | Best Configuration | Val Macro-F1 |
|---|---|---|
| Logistic Regression | `C=10`, L2, `class_weight='balanced'` | 0.8819 |
| Naive Bayes | `alpha=0.1` | 0.5933 |
| Random Forest | `n=100`, `depth=None` | 0.8696 |
| SimpleRNN | `units=128`, `lr=5e-4`, `dropout=0.3` | 0.5713 |
| GRU | `units=64`, `lr=1e-3` | 0.8745 |
| LSTM | `units=64`, `lr=1e-3` | 0.8674 |
| Bidirectional SimpleRNN | `units=64`, `lr=1e-3` | 0.7246 |
| Bidirectional GRU | `units=64`, `lr=1e-3` | 0.8800 |
| Bidirectional LSTM | `units=128`, `lr=1e-3` | 0.8736 |
| BERT Base | `lr=2e-5`, `bs=16`, epochs≤8 (early-stopped) | 0.6220 |

## 4. Results

### 4.1 Overall Test Performance

| Rank | Model | Test Accuracy | Test F1 (Macro) | Test F1 (Weighted) |
|---|---|---|---|---|
| 🏆 1 | **Logistic Regression** | **0.8886** | **0.8724** | **0.9172** |
| 2 | Bidirectional GRU | 0.8864 | 0.8691 | 0.9145 |
| 3 | GRU | 0.8807 | 0.8644 | 0.9101 |
| 4 | Bidirectional LSTM | 0.8783 | 0.8627 | 0.9070 |
| 5 | LSTM | 0.8750 | 0.8594 | 0.9048 |
| 6 | Random Forest | 0.8702 | 0.8505 | 0.8779 |
| 7 | Bidirectional SimpleRNN | 0.7429 | 0.7291 | 0.7791 |
| 8 | SimpleRNN | 0.6226 | 0.5920 | 0.6548 |
| 9 | BERT Base | 0.5543 | 0.5992 | 0.5926 |
| ❌ 10 | **Naive Bayes** | 0.5719 | **0.5775** | 0.6255 |

Three tiers emerge: a strong group (Logistic Regression, Bidirectional GRU, GRU, Bidirectional LSTM, LSTM, Random Forest); a mid-tier of Bidirectional SimpleRNN; and a bottom group of SimpleRNN, BERT Base, and Naive Bayes.

### 4.2 Best and Worst Models

**🏆 Best: class-balanced Logistic Regression with TF-IDF** (Macro-F1 = 0.8724). It slightly outperforms every neural system, likely because JailBreakV-28K's policy labels are tied to strong lexical cues, repeated attack templates, and category-specific phrasing that unigram/bigram TF-IDF features capture directly, sorted efficiently by a linear model in high-dimensional sparse space.

**Strongest neural model: Bidirectional GRU** (Macro-F1 = 0.8691), benefiting from bidirectional context over unidirectional GRU by a small margin; LSTM and Bidirectional LSTM show a similar, more modest pattern — gated recurrent memory helps, bidirectionality adds a smaller secondary benefit.

**❌ Worst: Naive Bayes** (Macro-F1 = 0.5775). Its conditional-independence assumption is too rigid for structured, multiword adversarial instructions. **BERT Base** also performs far below the best classical and recurrent models — not because transformers are inherently unsuited to jailbreak classification, but because the current data distribution and fine-tuning configuration favor lexical and recurrent representations instead (see Section 5).

### 4.3 Per-Class Performance (Best Model: Logistic Regression)

Precision and recall are high for most classes: `Malware` reaches F1 = 0.97, `Bias` and `Fraud` both reach 0.96, and `Animal Abuse`, `Violence`, `Economic Harm`, `Illegal Activity`, `Tailored Unlicensed Advice`, `Unethical Behavior`, and `Government Decision` all score strongly.

The exception is **`Child Abuse Content`**, which achieves very high recall but low precision — the model over-predicts this minority label, indicating lexical/template overlap with other categories. `Health Consultation`, `Privacy Violation`, and `Political Sensitivity` also score lower, likely from smaller support and higher semantic overlap with other classes.

### 4.4 Confusion Matrix Analysis

Logistic Regression classifies most large-category examples correctly, but several other-policy prompts are wrongly labeled `Child Abuse Content` — consistent with class imbalance, repeated templates, and overlapping lexical cues. Ranked by test accuracy, the Ensemble, Logistic Regression, and Bidirectional GRU show the strongest diagonal concentration, followed by Bidirectional LSTM and Random Forest; SimpleRNN, Naive Bayes, and BERT Base show much wider cross-class confusion.

### 4.5 Error Analysis

Logistic Regression misclassifies **468 of 4,200** test cases (**11.14% error rate**). The dominant error is `Illegal Activity` → `Child Abuse Content` (42 cases), followed by `Government Decision` (30), `Privacy Violation` (29), `Political Sensitivity` (29), `Malware` (29), `Unethical Behavior` (27), `Economic Harm` (25), `Physical Harm` (24), `Fraud` (23), and `Hate Speech` (23) — all misrouted into `Child Abuse Content`.

The confused prompt excerpts share similar template wording — generic instructions to complete numbered lists or give steps for an activity — supporting the EDA finding that repeated jailbreak wrappers can dominate the lexical signal and make prompts with different underlying harms look similar to the classifier.

### 4.6 Soft-Voting Ensemble

A soft-voting ensemble of the three best validation-selected models — **Logistic Regression, Bidirectional GRU, GRU** — averages their class-probability outputs:

| | Test Accuracy | Test F1 (Macro) | Test F1 (Weighted) |
|---|---|---|---|
| Ensemble (Top-3) | **0.8898** | **0.8733** | **0.9177** |

The gain over standalone Logistic Regression (0.8724 → 0.8733 Macro-F1) is small but meaningful: the constituents use different representations — sparse lexical (Logistic Regression) vs. sequential (GRU-based) — so their errors aren't fully correlated, letting the ensemble draw on complementary evidence without a heavier meta-model.

### 4.7 Input-Ablation Study

Concatenating `redteam_query` with `jailbreak_query` raises Logistic Regression's validation Macro-F1 from **0.8833 to 0.9910** — a far larger jump than any architectural or ensemble change produced. This is interpreted with caution: `redteam_query` reveals a cleaner version of the harmful intent that `policy` was likely derived from, information that would never be available to a real deployment-time guardrail. This result is reported as a methodological finding about the dataset, **not** used as a legitimate feature — `redteam_query` is excluded from all 10 primary models.

## 5. Discussion

Model complexity does not necessarily improve classification performance on JailBreakV-28K. TF-IDF + Logistic Regression slightly outperforms the best recurrent models and vastly outperforms BERT Base under the tested training conditions — consistent with the EDA finding that this dataset contains strong lexical regularities and repeated jailbreak templates built around common wrapper phrases ("will", "always", "anything", "respond") that sparse n-gram features capture efficiently without semantic reasoning.

BERT's underperformance (0.5543 accuracy, 0.5992 Macro-F1) is the most unexpected result, given its theoretical capacity for stronger contextual modeling. This is consistent with Mosbach et al.'s (2021) account of BERT fine-tuning instability: aggressive learning rates can cause representation collapse in pretrained lower layers, pushing the optimizer toward degenerate, near-majority-class predictions. Subword fragmentation of adversarial obfuscation, use of the pooler output rather than the `[CLS]` token, and early stopping on a stagnant validation loss were likely additional contributing factors; Sun et al. (2020) and Mosbach et al. (2021) suggest AdamW with a warmup schedule and a 2×10⁻⁵ learning rate could push BERT well above its current rank. Sequence truncation, class imbalance, and the limited (3-config) manual search likely also played a role.

Among recurrent models, GRU and LSTM massively outperform SimpleRNN (0.8644 / 0.8594 vs. 0.5920 Macro-F1), confirming the value of gated memory over long ranges. Bidirectional GRU achieves the best neural result (0.8691), hinting that late-prompt information helps interpret earlier content and vice versa — though the bidirectional benefit is smaller than the gain from gating itself.

The ensemble's modest improvement (0.8724 → 0.8733 Macro-F1) suggests lexical and recurrent representations capture largely overlapping signal. Finally, the ablation's near-perfect score with `redteam_query` included is a caution against interpreting large jumps in safety-classification performance as real-world gains: such leaps should always be checked for leakage before being reported as genuine improvement.

## 6. Conclusion

This study compares ten models across TF-IDF, Word2Vec, GloVe, and BERT-based representations for jailbreak harm categorization on the 28,000-sample JailBreakV-28K dataset. The strongest individual model was class-balanced Logistic Regression with TF-IDF (0.8886 accuracy, 0.8724 Macro-F1), followed by Bidirectional GRU (0.8691 Macro-F1); Naive Bayes was weakest. A soft-voting ensemble of Logistic Regression, BiGRU, and GRU improved results to 0.8898 accuracy and 0.8733 Macro-F1. When jailbreak categories carry strong word-level signals, basic lexical models can compete successfully with — and here, beat — much larger neural architectures. Performance should nonetheless be read with the caveat that the dataset's repetitive templates may limit generalization to novel jailbreak tactics.

## 7. Limitations

1. **Template overlap across splits.** Abundant similar/related `jailbreak_query` templates mean a random stratified split can place structurally similar prompts in both train and test, likely overestimating generalization to genuinely unseen attack styles. Future work should use template-disjoint or attack-family-disjoint splitting.
2. **Class imbalance.** Despite Macro-F1, class weighting, and stratified splitting, the smallest categories (`Child Abuse Content`, `Privacy Violation`, `Political Sensitivity`, `Health Consultation`) still show weaker per-class results.
3. **Limited transformer tuning.** BERT Base was evaluated over only 3 manual configurations as specified by the project; a broader search over learning rate, batch size, sequence length, layer freezing, warmup, and training duration would likely raise its performance. The reported BERT result reflects this specific training scenario, not an upper bound on transformer-based classification.
4. **Single benchmark.** All experiments are conducted on JailBreakV-28K alone; performance here does not guarantee robustness to new jailbreak techniques, other domains, or attacks devised after the dataset's collection.

## Ethical Considerations

The dataset contains malicious and hostile prompts. All experiments in this project are conducted for safety analysis, model evaluation, and defensive classification — the resulting classifier is intended to support safety review, not to enable harmful use, and this report avoids reproducing unnecessary harmful content. Model predictions should not substitute for human or policy-level oversight in real deployment, but rather serve as one component of a larger safety system.

## Repository Structure

```
Jailbreak-Attack-Prompt-Classification-Model/
├── notebooks/
│   └── Jailbreak_Harm_Category_Classification.ipynb   # Full pipeline: EDA -> preprocessing -> 10 models -> evaluation
├── artifacts/
│   └── results/            # all_runs.csv, final_test_metrics.csv
├── figures/                 # Confusion matrices, EDA plots, ranking charts
├── deployment/               # Exported inference pipeline (TF-IDF + Logistic Regression)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Reproducing Results

The notebook is self-installing (installs `gensim`, `wordcloud`, `datasets`, `transformers`, `accelerate` if missing) and downloads the dataset directly from HuggingFace Hub — no manual data setup required.

**Recommended: Google Colab or Kaggle (GPU runtime)**
1. Open `notebooks/Jailbreak_Harm_Category_Classification.ipynb` in [Colab](https://colab.research.google.com/) or [Kaggle](https://www.kaggle.com/) (Kaggle notebooks default to a T4 GPU).
2. Set the runtime/accelerator to GPU (`Runtime -> Change runtime type -> GPU` in Colab).
3. Run all cells top to bottom (`Runtime -> Run all`).
4. Outputs are written to `figures/`, `artifacts/results/`, and `deployment/` in the working directory.

**Local setup**
```bash
git clone https://github.com/Sadiabiscuit/Jailbreak-Attack-Prompt-Classification-Model.git
cd Jailbreak-Attack-Prompt-Classification-Model
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/Jailbreak_Harm_Category_Classification.ipynb
```
A CUDA-capable GPU is strongly recommended for the RNN and BERT phases; classical ML runs fine on CPU.

## Deployment Artifact

The notebook exports a lightweight, deployable inference pipeline (`deployment/`) built from the best-performing model overall — the fitted TF-IDF vectorizer + Logistic Regression model, label encoder, and preprocessing function — with a `metadata.json` describing the model and its test metrics (Accuracy 0.8886, Macro F1 0.8724).

## Tech Stack

`scikit-learn` · `TensorFlow / Keras` · `PyTorch` · `HuggingFace Transformers` & `Datasets` · `Gensim` · `NLTK` · `pandas` / `numpy` · `matplotlib` / `seaborn`

## References

1. Luo, W., Ma, S., Liu, X., Guo, X., & Xiao, C. (2024). JailBreakV-28K: A Benchmark for Assessing the Robustness of Multimodal Large Language Models against Jailbreak Attacks. *COLM*. https://arxiv.org/abs/2404.03027
2. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.
3. Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). Efficient estimation of word representations in vector space. https://doi.org/10.48550/arXiv.1301.3781
4. Pennington, J., Socher, R., & Manning, C. D. (2014). GloVe: Global vectors for word representation. *EMNLP*, 1532–1543.
5. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
6. Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder–decoder for statistical machine translation. *EMNLP*, 1724–1734.
7. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*, 4171–4186.
8. Mosbach, M., Andriushchenko, M., & Klakow, D. (2021). On the stability of fine-tuning BERT: Misconceptions, explanations, and strong baselines. https://doi.org/10.48550/arXiv.2006.04884
9. Sun, C., Qiu, X., Xu, Y., & Huang, X. (2020). How to fine-tune BERT for text classification? https://doi.org/10.48550/arXiv.1905.05583
10. Hawkins, J., Pramar, A., Beard, R., & Chandra, R. (2025). Machine learning for detection and analysis of novel LLM jailbreaks. https://doi.org/10.48550/arXiv.2510.01644

## License

This project is licensed under the [MIT License](LICENSE).

## Model Explanation video
https://youtu.be/KF5NQG5ZbPU?si=_myaaoY-Jz6JHQmt
## Contributors:
#Sadia Siddiqa
#Mahin Husayna
#SK. Azizul Karim
#Iftikar Rahman

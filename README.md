# Financial Sentiment Analysis

A machine learning project that classifies financial news sentences as **positive**, **negative**, or **neutral** sentiment.

## Dataset

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) by Malo et al. (2014)
- 4,838 sentences from financial news, annotated by finance professionals (after deduplication)
- 3 classes: neutral (~59%), positive (~28%), negative (~12%)

## Project Structure

```
notebooks/
├── 01-data-loading.ipynb     # data ingestion and initial inspection
├── 02-eda.ipynb              # exploratory data analysis
├── 03-preprocessing.ipynb    # text cleaning and feature engineering
├── 04-classical-ml.ipynb     # TF-IDF + Logistic Regression (GridSearchCV) + XGBoost (Optuna, 100 trials)
└── 05-finetuned-ml.ipynb     # FinBERT zero-shot evaluation and fine-tuning
data/
├── raw/                      # original source files
└── processed/                # train/valid/test splits + clean_df
visualization/                # saved confusion matrix plots
```

## Results

All metrics evaluated on the held-out test set.

| Model | Macro F1 | Weighted F1 | Accuracy |
|---|---|---|---|
| LogReg (GridSearchCV) | 0.642 | 0.701 | 0.701 |
| XGBoost (Optuna) | 0.619 | 0.692 | 0.707 |
| FinBERT zero-shot | 0.885 | 0.894 | 0.893 |
| FinBERT fine-tuned | **0.889** | **0.899** | **0.899** |

## Key Findings

- FinBERT zero-shot already outperforms classical ML by ~24 points in macro F1 — domain-specific pre-training matters a lot
- Fine-tuning adds a further ~0.4 points on top of zero-shot, showing the training data is useful but the base model is already strong
- Classical models struggle most with the minority class (negative, ~12% of data); FinBERT handles class imbalance much better

## How to Run

```bash
pip install -r requirements.txt  # or: poetry install
```

Run notebooks in order: `01` → `02` → `03` → `04` → `05`

> Note: notebook 05 requires a GPU for fine-tuning. To skip training, comment out `epoch_loop()` — the saved checkpoint will be loaded automatically.

## Tech Stack

- **Data:** `pandas`, `datasets` (HuggingFace)
- **EDA & visualization:** `matplotlib`, `seaborn`
- **NLP preprocessing:** `nltk`
- **Classical ML:** `scikit-learn`, `xgboost`, `optuna`
- **Transformers:** `transformers`, `torch` (FinBERT)

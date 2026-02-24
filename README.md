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
└── 04-classical-ml.ipynb     # classical ML models (TF-IDF + classifiers)
data/
├── raw/                      # original source files
└── processed/                # train/valid/test splits + clean_df
src/
├── data/
├── features/
├── models/
└── visualization/
```

## Tech Stack

- **Data:** `pandas`, `datasets` (HuggingFace)
- **EDA & visualization:** `matplotlib`, `seaborn`
- **NLP preprocessing:** `nltk`
- **Modeling:** `scikit-learn`, `xgboost`

# Financial Sentiment Analysis

A machine learning project that classifies financial news sentences as **positive**, **negative**, or **neutral** sentiment. Built as a portfolio project demonstrating the full ML pipeline — from raw data to a deployed web demo.

## Dataset

[Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) by Malo et al. (2014)
- 4,838 sentences from financial news, annotated by finance professionals (after deduplication)
- 3 classes: neutral (~59%), positive (~28%), negative (~12%)

## Project Structure

```
notebooks/
├── 01-data-loading.ipynb 
├── 02-eda.ipynb          
src/
├── data/                 
├── features/             
├── models/               
└── visualization/        
```

## Tech Stack

- **Data:** `pandas`, `datasets` (HuggingFace)
- **EDA:** `matplotlib`
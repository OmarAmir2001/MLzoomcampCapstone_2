# Two-Stage Hierarchical ML System for Customer Support Ticket Analysis

A production-ready machine learning system for predicting ticket types and priorities using a hierarchical two-stage approach.

## 🎯 Overview

This project implements a two-stage hierarchical classification system:

- **Stage 1**: Predicts `ticket_type` (e.g., Technical Issue, Billing Inquiry, Product Inquiry)
- **Stage 2**: Predicts `ticket_priority` (e.g., Critical, High, Medium, Low) using Stage 1 predictions

### Key Features

✅ **Modular Architecture**: Clean separation of preprocessing, training, and inference  
✅ **No Data Leakage**: Explicit exclusion of target-related features  
✅ **Class Imbalance Handling**: Balanced class weights in all models  
✅ **Production-Ready**: FastAPI/Docker deployment compatible  
✅ **Comprehensive Evaluation**: Macro F1-score, confusion matrices, classification reports

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0
- pyyaml >= 6.0
- joblib >= 1.2.0
- scipy >= 1.9.0

## 🚀 Quick Start

### 1. Train the Pipeline

```bash
python src/train_pipeline.py \
    --data data/customer_support_tickets.csv \
    --output models/ \
    --evaluate
```

This will:
- Train both Stage 1 and Stage 2 models
- Save models to `models/` directory
- Generate evaluation report with confusion matrices

### 2. Make Predictions

**Batch Prediction:**
```bash
python src/predict.py batch \
    --model models/ \
    --input data/new_tickets.csv \
    --output predictions.csv
```

**Interactive Mode:**
```bash
python src/predict.py interactive --model models/
```

## 📊 Architecture

### Feature Engineering

**Text Features:**
- TF-IDF vectorization on combined `ticket_subject + ticket_description`
- Max features: 5000
- N-gram range: (1, 2)

**Structured Features:**
- `product_purchased` (one-hot encoded)
- `ticket_channel` (one-hot encoded)
- `customer_gender` (one-hot encoded)
- `customer_age` (standardized)

**Excluded Features** (to prevent data leakage):
- `ticket_priority` (target for Stage 2)
- `ticket_status`
- `date_of_purchase`

### Two-Stage Process

```
┌─────────────────────────────────────────────────────────┐
│                     Input Ticket                        │
│  (subject, description, product, channel, gender, age)  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Text Preprocessing  │
          │     (TF-IDF)         │
          └──────────┬───────────┘
                     │
                     ▼
      ┌──────────────────────────────┐
      │ Structured Feature Encoding  │
      │    (One-Hot + Scaling)       │
      └──────────┬───────────────────┘
                 │
                 ▼
    ┌────────────────────────────┐
    │   Stage 1: Ticket Type     │
    │  (Logistic Regression)     │
    └────────┬───────────────────┘
             │
             │ Predicted Type
             ▼
┌────────────────────────────────────┐
│  Stage 2: Ticket Priority          │
│  (Uses predicted type + features)  │
│     (Logistic Regression)          │
└────────┬───────────────────────────┘
         │
         ▼
  ┌──────────────────┐
  │  Final Output    │
  │  Type + Priority │
  └──────────────────┘
```

## 💻 Usage Examples

### Python API

```python
from src.inference import load_pipeline, predict_single

# Load trained pipeline
predictor = load_pipeline('models/')

# Make prediction
ticket = {
    'ticket_subject': 'Product setup',
    'ticket_description': 'Having trouble setting up the device',
    'product_purchased': 'gopro_hero',
    'ticket_channel': 'email',
    'customer_gender': 'male',
    'customer_age': 35
}

result = predict_single(ticket, predictor)
print(f"Ticket Type: {result['ticket_type']}")
print(f"Ticket Priority: {result['ticket_priority']}")
print(f"Type Confidence: {result['ticket_type_confidence']:.2%}")
```

### Training with Custom Configuration

```python
from src.utils import load_data, split_data
from src.training import train_full_pipeline

# Load data
df = load_data('data/customer_support_tickets.csv')
df_train, df_val, df_test = split_data(df)

# Custom configuration
config = {
    'text_preprocessing': {
        'max_features': 3000,
        'ngram_range': [1, 2]
    },
    'stage1': {
        'model_type': 'logistic',
        'params': {'C': 0.5, 'max_iter': 1000}
    },
    'stage2': {
        'model_type': 'svm',
        'params': {'C': 1.0, 'max_iter': 1000}
    }
}

# Train
predictor, metrics = train_full_pipeline(df_train, df_val, config)
predictor.save('models/')
```

## 📈 Model Performance

Expected performance on validation set:

| Stage | Metric | Score |
|-------|--------|-------|
| Stage 1 (Ticket Type) | Macro F1 | > 0.60 |
| Stage 2 (Ticket Priority) | Macro F1 | > 0.55 |

## 🔍 Evaluation

Generate comprehensive evaluation report:

```bash
python src/train_pipeline.py \
    --data data/customer_support_tickets.csv \
    --output models/ \
    --evaluate
```

This creates:
- Confusion matrices for both stages
- Classification reports
- Per-class F1 scores
- Saved plots in `models/evaluation_report/`

## 📁 Project Structure

```
MLzoomcampCapstone_2/
├── data/
│   └── customer_support_tickets.csv
├── src/
│   ├── __init__.py
│   ├── preprocessing.py      # Feature engineering
│   ├── models.py             # Stage 1 & Stage 2 classifiers
│   ├── training.py           # Training pipelines
│   ├── evaluation.py         # Metrics and visualization
│   ├── inference.py          # Prediction functions
│   ├── utils.py              # Helper functions
│   ├── train_pipeline.py     # Training script
│   └── predict.py            # Prediction script
├── notebooks/
│   └── Eda_and_cleaning.ipynb
├── models/                   # Saved models (created after training)
├── requirements.txt
└── README.md
```

## 🛡️ Data Leakage Prevention

The implementation explicitly prevents data leakage:

1. **Excluded Features**: `ticket_priority`, `ticket_status`, `date_of_purchase` are never used as features
2. **Stage 2 Training**: Uses ground-truth `ticket_type` labels
3. **Stage 2 Inference**: Uses **predicted** `ticket_type` from Stage 1 (not ground truth)

## 🎓 Model Details

### Stage 1: Ticket Type Classifier

- **Model**: Logistic Regression (default) or Linear SVM
- **Features**: TF-IDF (5000 features) + Structured (one-hot + scaled)
- **Class Imbalance**: `class_weight='balanced'`
- **Regularization**: C=1.0 (tunable)

### Stage 2: Ticket Priority Classifier

- **Model**: Logistic Regression (default) or Linear SVM
- **Features**: Same as Stage 1 + **predicted ticket_type**
- **Class Imbalance**: `class_weight='balanced'`
- **Regularization**: C=1.0 (tunable)

## 🚢 Deployment

The modular architecture is designed for easy deployment:

```python
# FastAPI example
from fastapi import FastAPI
from src.inference import load_pipeline, predict_single

app = FastAPI()
predictor = load_pipeline('models/')

@app.post("/predict")
def predict(ticket: dict):
    return predict_single(ticket, predictor)
```

## 📝 License

MIT License

## 👥 Author

ML Zoomcamp Capstone Project
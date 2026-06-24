# Twitter Sentiment Analysis System

An intelligent multi-class Natural Language Processing (NLP) system that automatically classifies Twitter posts (tweets) into **Positive**, **Negative**, or **Neutral** categories. The application contains a complete preprocessing and training pipeline, a serialized best model, and an interactive Streamlit visualization dashboard.

---

## 📂 Project Structure

```
├── data/                      # Cached datasets (auto-downloaded)
├── models/                    # Serialized joblib pipelines and metrics json
│   ├── best_model.joblib      # Default trained SVM pipeline
│   ├── logistic_regression.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── metrics.json           # Evaluation metrics of all trained models
├── src/                       # Source Modules
│   ├── data_loader.py         # Downloads, filters, and loads Twitter data
│   ├── preprocessing.py      # Cleans, tokenizes, and lemmatizes text
│   ├── features.py            # configures Bag of Words & TF-IDF Vectorizers
│   ├── models.py              # Builds pipelines and computes performance stats
│   └── predict.py             # Inference class for real-time predictions
├── app.py                     # Streamlit visualization and prediction app
├── train.py                   # Model training entrypoint script
└── README.md                  # User documentation (this file)
```

---

## 🚀 Getting Started

### 1. Prerequisites & Installation

Ensure you have Python installed (recommended: Python 3.10+). Install the required packages:

```bash
pip install pandas numpy scikit-learn nltk plotly streamlit requests tqdm emoji joblib
```

### 2. Model Training & Comparison

Run the training script to fetch the Twitter Entity Sentiment Analysis dataset, clean and preprocess the tweets (lemmatization and stopword removal), train candidate models (Naive Bayes, Logistic Regression, and SVM), compare their accuracy on the validation set, and save the models:

```bash
python train.py
```

*Preprocessing results are cached inside `data/` after the first run to accelerate subsequent executions.*

### 3. Run the Streamlit Interactive Dashboard

Launch the Streamlit web application to view comparative model metrics, global/topic-level sentiment distributions, word frequencies, and test real-time tweet prediction:

```bash
streamlit run app.py
```

Once started, the dashboard will be available at: **`http://localhost:8501`**.

### 4. Command-Line Predictions

You can classify custom text directly from your shell using the inference module:

```bash
python src/predict.py "The new update is absolutely spectacular! I love it."
```

*Expected JSON Output:*
```json
{
  "sentiment": "Positive",
  "confidence": 0.81,
  "processed_text": "new update absolutely spectacular love"
}
```

---

## 📊 Model Performance Summary

Models evaluated on the **Twitter Entity Sentiment Analysis** validation dataset:

- **Support Vector Machine (SVM)**: **92.50% Accuracy** (Default Recommended Model)
- **Logistic Regression**: **92.02% Accuracy** (Highly recommended for general/out-of-corpus predictions)
- **Naive Bayes**: **83.92% Accuracy**

*All detailed metrics, including confusion matrices and classification reports, are visible in the **Model Benchmarks** tab of the dashboard.*

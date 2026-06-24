import os
import json
import time
import pandas as pd
from tqdm import tqdm
from src.data_loader import get_datasets, DATA_DIR
from src.preprocessing import preprocess_tweet
from src.features import get_vectorizer
from src.models import create_pipeline, train_and_evaluate_model, save_model_pipeline, MODELS_DIR

# Register tqdm with pandas
tqdm.pandas()

def main():
    print("=" * 60)
    print("Twitter Sentiment Analysis - Model Training Pipeline")
    print("=" * 60)
    
    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    df_train, df_val = get_datasets()
    
    # 2. Preprocess Text (with caching)
    preprocessed_train_path = os.path.join(DATA_DIR, "preprocessed_train.csv")
    preprocessed_val_path = os.path.join(DATA_DIR, "preprocessed_val.csv")
    
    if os.path.exists(preprocessed_train_path) and os.path.exists(preprocessed_val_path):
        print("\n[2/5] Loading preprocessed data from cache...")
        df_train = pd.read_csv(preprocessed_train_path)
        df_val = pd.read_csv(preprocessed_val_path)
        # Handle any nulls that might arise from empty cleaned strings
        df_train["cleaned_text"] = df_train["cleaned_text"].fillna("")
        df_val["cleaned_text"] = df_val["cleaned_text"].fillna("")
    else:
        print("\n[2/5] Preprocessing text data (cleaning, tokenizing, lemmatizing)...")
        print("Cleaning training set...")
        df_train["cleaned_text"] = df_train["tweet_content"].progress_apply(preprocess_tweet)
        print("Cleaning validation set...")
        df_val["cleaned_text"] = df_val["tweet_content"].progress_apply(preprocess_tweet)
        
        # Save preprocessed cache
        df_train.to_csv(preprocessed_train_path, index=False)
        df_val.to_csv(preprocessed_val_path, index=False)
        print("Preprocessed data cached successfully.")
        
    # Remove any rows with empty cleaned_text
    df_train = df_train[df_train["cleaned_text"] != ""]
    df_val = df_val[df_val["cleaned_text"] != ""]
    
    X_train, y_train = df_train["cleaned_text"], df_train["sentiment"]
    X_val, y_val = df_val["cleaned_text"], df_val["sentiment"]
    
    print(f"Train samples after preprocessing: {len(X_train)}")
    print(f"Val samples after preprocessing: {len(X_val)}")
    
    # 3. Model Training & Comparison
    print("\n[3/5] Setting up text vectorizer...")
    vectorizer = get_vectorizer(method="tfidf", max_features=15000, ngram_range=(1, 2))
    
    candidate_models = ["naive_bayes", "logistic_regression", "svm"]
    results = {}
    trained_pipelines = {}
    
    print("\n[4/5] Training and evaluating candidate models...")
    for model_name in candidate_models:
        print(f"\n--- Training {model_name} ---")
        start_time = time.time()
        
        pipeline = create_pipeline(vectorizer, model_type=model_name)
        pipeline, metrics = train_and_evaluate_model(pipeline, X_train, y_train, X_val, y_val)
        
        elapsed = time.time() - start_time
        metrics["training_time_sec"] = elapsed
        print(f"{model_name} Training Completed in {elapsed:.2f}s")
        print(f"Accuracy: {metrics['accuracy']:.4f} | F1-Score: {metrics['f1_score']:.4f}")
        
        results[model_name] = metrics
        trained_pipelines[model_name] = pipeline
        
        # Save candidate model
        save_model_pipeline(pipeline, model_name)
        
    # 4. Compare & Save Best Model
    print("\n[5/5] Selecting the best model...")
    best_model_name = max(results, key=lambda k: results[k]["accuracy"])
    best_accuracy = results[best_model_name]["accuracy"]
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_accuracy:.4f}")
    
    best_pipeline = trained_pipelines[best_model_name]
    save_model_pipeline(best_pipeline, "best_model")
    
    # Save training metrics to JSON for the Streamlit dashboard
    metrics_summary = {
        "best_model": best_model_name,
        "results": results,
        "class_distribution": {
            "train": df_train["sentiment"].value_counts().to_dict(),
            "val": df_val["sentiment"].value_counts().to_dict()
        }
    }
    
    metrics_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print(f"Saved evaluation metrics metadata to {metrics_path}")
    print("\nPipeline training completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()

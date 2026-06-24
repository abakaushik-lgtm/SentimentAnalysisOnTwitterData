import os
import pandas as pd
import requests

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "twitter_training.csv")
VAL_PATH = os.path.join(DATA_DIR, "twitter_validation.csv")

TRAIN_URL = "https://raw.githubusercontent.com/11Shraddha/SentimentAnalysis_NLP/main/twitter_training.csv"
VAL_URL = "https://raw.githubusercontent.com/11Shraddha/SentimentAnalysis_NLP/main/twitter_validation.csv"

def download_file(url: str, dest_path: str):
    """Downloads a file from a URL to a destination path if it doesn't exist."""
    if os.path.exists(dest_path):
        return
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Downloading dataset from {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")

def load_and_clean_data(file_path: str, url: str) -> pd.DataFrame:
    """
    Downloads, loads, and preprocesses the dataset:
    - Sets column names
    - Filters to Positive, Negative, and Neutral sentiments
    - Drops missing text values
    """
    download_file(url, file_path)
    
    cols = ["tweet_id", "entity", "sentiment", "tweet_content"]
    df = pd.read_csv(file_path, names=cols, header=None)
    
    # Drop rows with missing text
    df = df.dropna(subset=["tweet_content"])
    
    # Normalize sentiment capitalization and filter classes
    df["sentiment"] = df["sentiment"].str.strip().str.capitalize()
    valid_sentiments = ["Positive", "Negative", "Neutral"]
    df = df[df["sentiment"].isin(valid_sentiments)]
    
    return df

def get_datasets():
    """Returns the training and validation DataFrames."""
    df_train = load_and_clean_data(TRAIN_PATH, TRAIN_URL)
    df_val = load_and_clean_data(VAL_PATH, VAL_URL)
    return df_train, df_val

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def get_vectorizer(method: str = "tfidf", max_features: int = 10000, ngram_range: tuple = (1, 2)):
    """
    Returns a text vectorizer:
    - tfidf: TF-IDF Vectorizer
    - bow: Bag of Words (Count) Vectorizer
    """
    if method.lower() == "tfidf":
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
    elif method.lower() == "bow":
        return CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.95
        )
    else:
        raise ValueError(f"Unknown vectorization method: {method}")

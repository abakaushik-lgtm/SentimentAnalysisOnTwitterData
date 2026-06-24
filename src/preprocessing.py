import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import emoji

# Ensure necessary NLTK downloads are available
# (The training entrypoint will run download verification)
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

try:
    LEMMATIZER = WordNetLemmatizer()
except LookupError:
    nltk.download("wordnet", quiet=True)
    LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """
    Cleans raw tweet text by:
    - Converting to lowercase
    - Removing URLs
    - Removing user mentions (@username)
    - Removing hashtags (#topic -> topic)
    - Removing emojis
    - Removing special characters, punctuation, and numbers
    - Removing extra whitespaces
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    
    # 3. Remove user mentions (@user)
    text = re.sub(r"@\w+", "", text)
    
    # 4. Remove emojis (using the emoji library if possible, otherwise regex)
    try:
        text = emoji.replace_emoji(text, replace="")
    except Exception:
        # Fallback regex for non-ascii (like emojis)
        text = re.sub(r"[^\x00-\x7F]+", "", text)
        
    # 5. Clean hashtags (keep the word, remove the '#')
    text = re.sub(r"#(\w+)", r"\1", text)
    
    # 6. Remove special characters, numbers and punctuation
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    
    # 7. Strip extra whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess_tweet(text: str) -> str:
    """
    Applies the full preprocessing pipeline:
    - Cleaning
    - Tokenization
    - Stopword removal
    - Lemmatization
    """
    # Clean the text
    cleaned = clean_text(text)
    if not cleaned:
        return ""
    
    # Tokenize
    try:
        tokens = word_tokenize(cleaned)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        tokens = word_tokenize(cleaned)
        
    # Remove stopwords and lemmatize
    processed_tokens = []
    for token in tokens:
        if token not in STOPWORDS:
            # Lemmatize word
            lemma = LEMMATIZER.lemmatize(token)
            processed_tokens.append(lemma)
            
    # Join back into a string
    return " ".join(processed_tokens)

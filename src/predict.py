import os
import sys
import numpy as np

# Adjust path for direct script executions
if __name__ == "__main__" and not __package__:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)

from src.preprocessing import preprocess_tweet
from src.models import load_model_pipeline

class SentimentPredictor:
    def __init__(self, model_name: str = "best_model"):
        """
        Initializes the predictor by loading the trained pipeline.
        """
        self.pipeline = load_model_pipeline(model_name)
        self.classes = self.pipeline.classes_
        
    def predict(self, text: str) -> dict:
        """
        Predicts the sentiment and confidence score of a raw input text.
        Returns:
            dict: {
                "sentiment": str,
                "confidence": float,
                "processed_text": str
            }
        """
        # Preprocess text
        cleaned_text = preprocess_tweet(text)
        
        # If text is empty after preprocessing, fallback to neutral with low confidence
        if not cleaned_text.strip():
            return {
                "sentiment": "Neutral",
                "confidence": 0.3333,
                "processed_text": "",
                "tokens": [],
                "probabilities": {"Positive": 0.3333, "Neutral": 0.3334, "Negative": 0.3333}
            }
            
        # Get prediction
        prediction = self.pipeline.predict([cleaned_text])[0]
        
        # Get prediction probabilities for confidence score
        prob_dict = {}
        try:
            probabilities = self.pipeline.predict_proba([cleaned_text])[0]
            class_idx = np.where(self.classes == prediction)[0][0]
            confidence = float(probabilities[class_idx])
            prob_dict = {self.classes[i]: round(float(probabilities[i]), 4) for i in range(len(self.classes))}
        except Exception:
            # Fallback if probability prediction is not supported
            confidence = 1.0
            prob_dict = {prediction: 1.0}
            
        return {
            "sentiment": prediction,
            "confidence": round(confidence, 4),
            "processed_text": cleaned_text,
            "tokens": cleaned_text.split(),
            "probabilities": prob_dict
        }

# Command-line utility to run manual tests
if __name__ == "__main__":
    import sys
    print("Initializing Sentiment Predictor...")
    try:
        predictor = SentimentPredictor()
        
        # Check if text was passed as argument
        if len(sys.argv) > 1:
            input_text = " ".join(sys.argv[1:])
        else:
            input_text = "The new smartphone update is amazing! Performance is much faster."
            
        print(f"\nInput: '{input_text}'")
        result = predictor.predict(input_text)
        print("Prediction Result:")
        print(result)
        
    except Exception as e:
        print("Error initializing predictor. Has the model been trained and saved?", e)

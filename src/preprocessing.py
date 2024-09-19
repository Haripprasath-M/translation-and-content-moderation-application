# src/preprocessing.py

import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import textstat
import nltk

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

important_stopwords = [
    "I", "you", "he", "she", "we", "they",
    "not", "no", "never", "can't", "won't", 
    "don't", "didn't", "hasn't", "haven't", "isn't", 
    "aren't", "wasn't", "weren't",
    "this", "that", "these", "those",
    "on", "in", "at", "for", "with",
    "and", "or", "but",
    "is", "are", "was", "were",
    "very", "too"
]

custom_stopwords = set(word for word in stop_words if word not in important_stopwords)

def clean_data(text):
    """Clean and preprocess the input text."""
    text = text.lower()  
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in custom_stopwords]
    return ' '.join(words)

def preprocess_comment(comment: str, scaler):
    """Preprocess the comment and extract features."""
    
    # Clean the comment text
    clean_text = clean_data(comment)

    # Check if cleaned text is empty
    if clean_text == '':
        return None  # Return None or handle as needed

    # Calculate additional features
    text_length = len(comment)  # Get the length of the comment

    # Perform sentiment analysis
    sentiment_polarity = TextBlob(comment).sentiment.polarity
    sentiment_subjectivity = TextBlob(comment).sentiment.subjectivity

    # Calculate readability scores
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(comment)
    gunning_fog = textstat.gunning_fog(comment)

    # Prepare a feature array for scaling
    features_to_scale = np.array([[text_length, sentiment_polarity, 
                                   sentiment_subjectivity, flesch_kincaid_grade, 
                                   gunning_fog]])

    # Scale the features using the scaler
    scaled_features = scaler.transform(features_to_scale)

    # Create a dictionary of features to return
    features = {
        'clean_text': clean_text,
        'text_length': scaled_features[0][0],
        'sentiment_polarity': scaled_features[0][1],
        'sentiment_subjectivity': scaled_features[0][2],
        'flesch_kincaid_grade': scaled_features[0][3],
        'gunning_fog': scaled_features[0][4],
    }

    return features
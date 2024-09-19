# src/model.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import KeyedVectors
import pickle
from preprocessing import clean_data

# Load the FastText model
def load_fasttext_model(model_path):
    return KeyedVectors.load(model_path)

# Load the LSTM model
def load_lstm_model(model_path):
    return load_model(model_path)

#Load Tokenizer
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as handle:
        return pickle.load(handle)
    
#Load Standard Scaler
def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as handle:
        return pickle.load(handle)

# Preprocess input text for prediction
def preprocess_input_text(tokenizer, fasttext_model, text, max_length=500):
    # Clean the input text 
    clean_text = clean_data(text) 

    # Convert text to sequences
    sequences = tokenizer.texts_to_sequences([clean_text])
    
    # Pad sequences to ensure uniform length
    padded_sequences = pad_sequences(sequences, maxlen=max_length)

    # Create embedding matrix for the input text based on the FastText model
    embedding_dim = fasttext_model.vector_size
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, i in tokenizer.word_index.items():
        if word in fasttext_model:
            embedding_matrix[i] = fasttext_model[word]
        else:
            embedding_matrix[i] = np.zeros(embedding_dim)

    return padded_sequences, embedding_matrix

# Make predictions using the LSTM model
def predict_with_lstm(lstm_model, tokenizer, fasttext_model, text):
    padded_sequences, embedding_matrix = preprocess_input_text(tokenizer, fasttext_model, text)
    
    prediction = lstm_model.predict(padded_sequences)
    
    # Convert probabilities to binary (0 or 1)
    prediction_binary = (prediction > 0.5).astype(int)
    
    return prediction_binary[0]  # Return the first prediction result

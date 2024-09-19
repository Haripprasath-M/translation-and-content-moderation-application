# src/fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing import preprocess_comment
from models import load_fasttext_model, load_lstm_model, \
load_tokenizer, load_scaler, predict_with_lstm

app = FastAPI()

# Define input model for API request
class Comment(BaseModel):
    text: str
    
# Load models when app starts
fasttext_model_path = 'src/models/fasttext-wiki-news-subwords-300.model'
lstm_model_path = 'src/models/lstm.h5'
tokenizer_path = 'src/models/tokenizer.pkl'
scaler_path = 'src/models/scaler.pkl'

fasttext_model = load_fasttext_model(fasttext_model_path)
lstm_model = load_lstm_model(lstm_model_path)
tokenizer = load_tokenizer(tokenizer_path)
scaler = load_scaler(scaler_path)


@app.post("/moderate/")
async def moderate_comment(comment: Comment):
    """Endpoint to moderate a comment."""
    
    features = preprocess_comment(comment.text, scaler)
    
    if features is None:
        return {"error": "Comment is empty after cleaning."}
    
    # Make predictions using LSTM model
    prediction = predict_with_lstm(lstm_model, tokenizer, fasttext_model, comment.text)
    
    # Define labels corresponding to the prediction outputs
    labels = [
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate"
    ]
    
    # Create a result dictionary to hold label information
    result = {}
    
    for i, pred in enumerate(prediction):
        if pred == 1:
            result[labels[i]] = "Yes"
        else:
            result[labels[i]] = "No"
    
    # Check if all predictions are 0 (not toxic)
    if all(value == "No" for value in result.values()):
        result["overall"] = "Appropriate"
    else:
        result["overall"] = "Inappropriate"
    
    return {
        "features": features,
        "prediction": result
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
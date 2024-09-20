# src/fastapi_app.py

from fastapi import FastAPI
from pydantic import BaseModel
from preprocessing import preprocess_comment
from models import load_fasttext_model, load_lstm_model, \
load_tokenizer, load_scaler, predict_with_lstm, load_translation_model,\
load_translation_tokenizer, translate_text

app = FastAPI()

# Define inputs model for API request
class Comment(BaseModel):
    text: str

class Message(BaseModel):
    text: str
    
# Load models when app starts
fasttext_model_path = 'src/models/fasttext-wiki-news-subwords-300.model'
lstm_model_path = 'src/models/lstm.h5'
tokenizer_path = 'src/models/tokenizer.pkl'
scaler_path = 'src/models/scaler.pkl'
translation_model_path = 'src/models/translation_model'
translation_tokenizer_path = 'src/models/translation_tokenizer'

fasttext_model = load_fasttext_model(fasttext_model_path)
lstm_model = load_lstm_model(lstm_model_path)
tokenizer = load_tokenizer(tokenizer_path)
scaler = load_scaler(scaler_path)
translation_model = load_translation_model(translation_model_path)
translation_tokenizer = load_translation_tokenizer(translation_tokenizer_path)

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

@app.post('/translate/')
async def translate(message: Message):
    """Endpoint to translate a message after checking for appropriateness."""
    
    #Moderate text
    moderation_result = await moderate_comment(Comment(text=message.text))
    
    if moderation_result["prediction"]["overall"] == "Inappropriate":
        return {"response": "Message is inappropriate and cannot be translated."}
    
    # If appropriate, perform translation
    translated_text = translate_text(translation_tokenizer, translation_model, message.text)
    
    return {"response": translated_text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=3000)
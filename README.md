**Translation and Content Moderation Application**

## Project Overview

This project builds a full-stack application that translates English text to French while maintaining content safety. It leverages FastAPI for the backend and Streamlit for the user-friendly frontend. A deep learning model acts as a content moderator, preventing the translation of inappropriate content.

## Data

- **Source**: Jigsaw Toxic Comment Classification Challenge (Kaggle)
- **Features**: Text data (`comment_text`) with labels for toxicity, hate speech, threats, etc.
- **Translation Model**: No separate dataset used; a pre-trained transformer model is employed.

## Exploratory Data Analysis (EDA)

Performed a comprehensive EDA to understand the comment data:

- Identified null and duplicate values.
- Generated descriptive statistics.
- Visualized class imbalance using count plots.
- Analyzed text length distribution with histograms.
- Explored feature correlations using a heatmap.
- Created word clouds to identify content patterns within each toxicity class.

## Data Preprocessing

- Cleaned the text data:
  - Lowercased characters.
  - Removed special characters and stopwords.
  - Applied lemmatization.
- Filtered out comments exceeding 500 characters.
- Enriched the data with:
  - Text length.
  - Sentiment polarity and subjectivity scores.
  - Readability metrics.
- Split the data into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).
- Standardized numerical features (text length, sentiment, and readability).

## Machine Learning Model

- **Input Features**: Cleaned text (TF-IDF vectors), text length, sentiment, and readability metrics.
- **Models**: Random Forest, Gradient Boosting, AdaBoost with OneVsRestClassifier for multi-label classification.
- **Evaluation**: F1-score and accuracy for each model.

**Model Performances:**

| Model        | Accuracy | F1-Score |
|--------------|----------|-----------|
| Random Forest | 0.906     | 0.637     |
| Gradient Boosting | 0.906     | 0.637     |
| AdaBoost      | 0.904     | 0.647     |

## Deep Learning Model

- Utilized the pre-trained FastText Wiki Subwords 300 model for word embeddings.
- Built and trained an LSTM model for content moderation, saving it as `lstm.h5`.

**LSTM Model Performance:**

- Accuracy: 0.916
- F1-Score: 0.755

## Machine Translation

- Employed a pre-trained transformer model `Helsinki-NLP/opus-mt-en-fr` for English-to-French translation.
- Saved a tokenizer and the model locally for text encoding and decoding.

## Challenges

- **Embedding Matrix**: FastText couldn't recognize numbers, resulting in zero vectors.
- **Model Storage**: Large pre-trained models are hosted on OneDrive due to GitHub storage limitations.
- **Docker**: Attempted implementation but storage constraints prevented completion.

## Deployment (Local)

- Not deployed to a server yet. The project includes:
  - A FastAPI backend with endpoints for content validation and translation.
  - A Streamlit frontend offering a user-friendly interface for text input, validation, and translation.

**To run locally:**

1. Clone the repository:

   ```bash
   git clone https://github.com/Haripprasath-M/translation-and-content-moderation-application
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the models from OneDrive and place them in the `models/` directory.
OneDrive link - https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL2YvYy84ZDc0M2QzMGMzYzMzZGZmL0VrUlVqLWc4amZsRG5TR2JjODlnUDdJQmVpcVVmSmFTazNZMnVSUmRmbzNsWUE%5FZT1mR1VKZ2M&id=8D743D30C3C33DFF%21se88f54448d3c43f99d219b73cf603fb2&cid=8D743D30C3C33DFF

4. Run the FastAPI backend:

   ```bash
   uvicorn src.fastapi_app:app --reload
   ```

5. Run the Streamlit frontend:

   ```bash
   streamlit run src/streamlit_app.py
   ```

## Future Work

- Complete Docker implementation for deployment.
- Further optimize the deep learning model.
- Add more translation options

## Contributions

Developed entirely by Haripprasath M.

## License

This project is open-source under the MIT License.

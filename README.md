# 😃 Emotion Detection from Text

## 📌 Project Overview

This project implements a **text-based emotion detection system** using machine learning. The goal is to classify textual inputs (such as tweets, sentences, or messages) into corresponding emotional categories like *Happy, Sad, Angry, Fear, Surprise,* and *Neutral*.

---

## 🧠 Technologies Used

- Python 🐍
- Scikit-learn
- Pandas
- NumPy
- NLTK
- Neattext
- Joblib

---

## 📂 Project Structure
```
EmotionDetection/
│
├── emotion_dataset.csv # Dataset used for training
├── Untitled.ipynb # Main Jupyter Notebook (model building & evaluation)
├── text_emotion_model.pkl # Saved trained model
├── tfidf_vectorizer.pkl # Saved TF-IDF vectorizer
├── label_encoder.pkl # Saved LabelEncoder (if used)
├── README.md # Project documentation

```

---

## 🔍 Dataset

- The dataset contains sentences/phrases labeled with one of several emotional states.
- Example columns:
  - `text`: The input text
  - `emotion`: The associated emotion label

---

## 🚀 How to Run the Project

1. **Clone the repository** (if hosted):
   ```bash
   git clone https://github.com/Rohit131101/EMOTIONDETECTION.git
   cd emotion-detection
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
If requirements.txt is not available, install manually:

bash
Copy
Edit
pip install pandas numpy scikit-learn nltk neattext joblib
Run the Jupyter notebook:
Open Untitled.ipynb in Jupyter or Google Colab and run all cells to train the model.

🧪 Sample Inference
python
Copy
Edit
import joblib

# Load model and vectorizer
model = joblib.load("text_emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Predict new text
text = ["I am feeling excited about life!"]
text_clean = [nfx.remove_stopwords(nfx.remove_punctuations(text[0]))]
text_vectorized = vectorizer.transform(text_clean)
prediction = model.predict(text_vectorized)

print("Predicted Emotion:", prediction[0])
✅ Model Performance
Model: Logistic Regression

Evaluation metrics used:

Classification report

Confusion matrix

Accuracy may vary based on train-test split and preprocessing steps.

📦 Future Improvements
Use deep learning models like LSTM or BERT for better accuracy

Build a REST API or Web App using Flask or Streamlit

Deploy to cloud (e.g., Heroku, AWS, etc.)


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from pathlib import Path

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

def classifier(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = str(text).lower()                               # lowercase
        text = re.sub(r'http\S+|www.\S+', '', text)            # remove urls
        text = re.sub(r'[^a-zA-Z\s]', '', text)                # remove numbers/punctuations
        tokens = word_tokenize(text)                           # tokenize
        tokens = [w for w in tokens if w not in stop_words]    # remove stopwords
        tokens = [lemmatizer.lemmatize(w) for w in tokens]     # lemmatize
        return " ".join(tokens)

    # Load trained model + vectorizer
    base_dir = Path(__file__).resolve().parent
    model_dir = base_dir / "Models" / "text_classifier"
    model_path = model_dir / "phishing_model.pkl"
    vectorizer_path = model_dir / "tfidf_vectorizer.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    if not vectorizer_path.exists():
        raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    text = clean_text(text)
    X_text = vectorizer.transform([text])
    prediction = model.predict(X_text)

    if prediction == 1:
        return "Phishing"
    else:
        return "Clean"

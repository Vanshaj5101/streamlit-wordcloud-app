import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text_series):
    """
    Cleans, removes stopwords, and lemmatizes comments from a Pandas Series.
    Returns a single concatenated cleaned string.
    """
    combined = " ".join(str(comment) for comment in text_series.dropna())

    # Remove non-alphabetic characters
    cleaned = re.sub(r"[^a-zA-Z\s]", "", combined)
    # Lowercase
    cleaned = cleaned.lower()
    # Tokenize
    words = cleaned.split()
    # Remove stopwords and lemmatize
    processed_words = [
        lemmatizer.lemmatize(word) for word in words if word not in stop_words
    ]

    return " ".join(processed_words)

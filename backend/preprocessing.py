import re 
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

stop_word = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() #Lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) #Remove punctuation and special characters
    tokens = word_tokenize(text) #Tokenize
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_word] #Remove stopwords and lemmatize
    return tokens

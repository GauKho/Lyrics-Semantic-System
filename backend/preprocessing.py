import re 
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punk_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

with open("backend\slang_map.json", "r") as f:
    slang_dict = json.load(f)

stop_word = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower() #Lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text) #Remove punctuation and special characters
    text = re.sub(r"\s+", " ", text)
    tokens = word_tokenize(text) #Tokenize

    tokens = [slang_dict.get(word, word) for word in tokens]

    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_word] #Remove stopwords and lemmatize
    return tokens

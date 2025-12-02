# detector/model_loader.py
import joblib
import pandas as pd
import re
from pathlib import Path
from transformers import MarianMTModel, MarianTokenizer

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent  # <-- CORRECTED: Points to 'detector'
ARTIFACTS_DIR = BASE_DIR / 'artifacts'      # <-- Points to 'detector/artifacts'
MODEL_PATH = ARTIFACTS_DIR / 'best_model.joblib'
VECTORIZER_PATH = ARTIFACTS_DIR / 'vectorizer.joblib'
TRANSLATION_MODEL_NAME = 'Helsinki-NLP/opus-mt-ar-en'

# Global variables to hold loaded models
BEST_MODEL = None
VECTORIZER = None
TRANSLATOR_TOKENIZER = None
TRANSLATOR_MODEL = None

def clean_english_text(text):
    """Clean and normalize English text, replicating the training step."""
    if pd.isna(text) or not text:
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    text = text.lower()
    text = ' '.join(text.split()).strip()
    return text

def translate_arabic_to_english(arabic_text):
    """Translate Arabic text to English using the loaded MarianMT model."""
    if not arabic_text or not TRANSLATOR_MODEL:
        return ""
    
    # Use 'cpu' for generating tokens to avoid CUDA requirement in simple deployments
    encoded_text = TRANSLATOR_TOKENIZER.prepare_seq2seq_batch([arabic_text], return_tensors="pt")
    translated_tokens = TRANSLATOR_MODEL.generate(**encoded_text)
    translated_text = TRANSLATOR_TOKENIZER.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

def load_models():
    """Loads all machine learning and translation models."""
    global BEST_MODEL, VECTORIZER, TRANSLATOR_MODEL, TRANSLATOR_TOKENIZER
    
    print("⏳ Loading ML and Translation Models...")
    
    try:
        # Load the trained ML Model (SVC) and Vectorizer (TFIDF)
        BEST_MODEL = joblib.load(MODEL_PATH)
        VECTORIZER = joblib.load(VECTORIZER_PATH)
        print("✅ ML Model (SVC) and Vectorizer loaded.")
    except Exception as e:
        print(f"🛑 ERROR loading ML artifacts: {e}")
        
    try:
        # Load Translation Model and Tokenizer
        TRANSLATOR_TOKENIZER = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
        TRANSLATOR_MODEL = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME)
        print(f"✅ Translation model {TRANSLATION_MODEL_NAME} loaded.")
    except Exception as e:
        print(f"🛑 ERROR loading translation model: {e}")

# Call the loader when Django initializes the app
load_models()
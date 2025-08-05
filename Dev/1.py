import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')          # Sentence + word tokenizer
nltk.download('stopwords')      # Common English stopwords
nltk.download('wordnet')        # Lemmatization dictionary
nltk.download('omw-1.4')        # Lemmatizer multilingual support
nltk.download('averaged_perceptron_tagger')  # Optional, for POS tagging
nltk.download('punkt_tab')
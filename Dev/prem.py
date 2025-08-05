import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the dataset
df = pd.read_csv("fake_or_real_news.csv")

# Initial inspection
print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())
print(df.info())
print("Label distribution:\n", df['label'].value_counts()) 

# Check for missing values in all columns
print("\nMissing values per column:\n", df.isnull().sum())

# Define cleaning function (but does not remove anything yet)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|[^a-zA-Z]", " ", str(text).lower())
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    return " ".join(tokens)

# Apply cleaning (temporary for inspection only)
df['clean_text'] = df['text'].apply(clean_text)

# Identify rows with empty or whitespace-only clean_text
empty_clean_rows = df[df['clean_text'].str.strip() == '']
print(f"\nRows with empty clean_text after preprocessing: {len(empty_clean_rows)}")
print(empty_clean_rows.head())

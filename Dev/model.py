import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import joblib
import re

# Create models subfolder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load data
df = pd.read_csv('fake_or_real_news_with_metadata.csv')

# Drop unnamed index column if exists
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Fill missing text with empty string
df['text'] = df['text'].fillna('')
df['title'] = df['title'].fillna('')

# Simulate domain extraction from title
def extract_domain(text):
    domain_patterns = ['cnn', 'fox', 'bbc', 'nbc', 'nyt', 'reuters', 'aljazeera', 'guardian', 'huffpost', 'abc']
    if not isinstance(text, str):
        return 'unknown'
    text_lower = text.lower()
    for keyword in domain_patterns:
        if keyword in text_lower:
            return keyword
    return 'other'

df['domain'] = df['title'].apply(extract_domain)

# Encode domain as one-hot
domain_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
domain_encoded = domain_ohe.fit_transform(df[['domain']])

# Convert label to binary
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# Split into train and test sets
X_train_text, X_test_text, y_train, y_test, X_train_domain, X_test_domain = train_test_split(
    df['text'], df['label'], domain_encoded, test_size=0.2, random_state=42, stratify=df['label']
)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Combine TF-IDF and domain features
X_train_combined = hstack([X_train_tfidf, X_train_domain])
X_test_combined = hstack([X_test_tfidf, X_test_domain])

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_combined, y_train)

# Predict
y_pred = model.predict(X_test_combined)

# Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# Save model and vectorizer in models/ subfolder
joblib.dump(model, 'models/logistic_model.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
joblib.dump(domain_ohe, 'models/domain_encoder.pkl')

print("Model and encoders saved successfully to 'models/' folder.")

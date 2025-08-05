import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib

# === Create models subfolder if it doesn't exist ===
os.makedirs('models', exist_ok=True)

# === Load data ===
df = pd.read_csv('fake_or_real_news_with_metadata.csv')

# Drop unnamed column
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Fill missing text and title
df['text'] = df['text'].fillna('')
df['title'] = df['title'].fillna('')

# Extract domain from title
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

# Encode domain
domain_ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
domain_encoded = domain_ohe.fit_transform(df[['domain']])

# Encode label
df['label'] = df['label'].map({'FAKE': 0, 'REAL': 1})

# === Split data ===
X_train_text, X_test_text, y_train, y_test, X_train_domain, X_test_domain = train_test_split(
    df['text'], df['label'], domain_encoded, test_size=0.2, random_state=42, stratify=df['label']
)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_text)
X_test_tfidf = tfidf_vectorizer.transform(X_test_text)

# Combine text + metadata
X_train_combined = hstack([X_train_tfidf, X_train_domain])
X_test_combined = hstack([X_test_tfidf, X_test_domain])

# === Model tuning using GridSearchCV ===
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train_combined, y_train)

print("Best Parameters:", grid.best_params_)

# Evaluate best model
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_combined)

print("\nClassification Report (Tuned Model):")
print(classification_report(y_test, y_pred, target_names=['FAKE', 'REAL']))

# === Save tuned model and encoders ===
joblib.dump(best_model, 'models/logistic_model_tuned.pkl')
joblib.dump(tfidf_vectorizer, 'models/tfidf_vectorizer_tuned.pkl')
joblib.dump(domain_ohe, 'models/domain_encoder_tuned.pkl')

print("Tuned model and encoders saved successfully in 'models/' directory.")

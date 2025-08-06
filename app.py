import streamlit as st
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
import pandas as pd
from textblob import TextBlob
import re
import matplotlib.pyplot as plt

# Load models and encoders (Not Tuned Ver.)
model_path = "models/logistic_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"
encoder_path = "models/domain_encoder.pkl"

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)
encoder = joblib.load(encoder_path)

# App title
st.set_page_config(page_title="🕵️‍♂️ Fake News Detector", layout="centered")
st.title("📰 Fake News Detection App")
st.subheader("Powered by Logistic Regression, TF-IDF, and Domain Features")

st.markdown("---")

# Input
title = st.text_input("🔤 Enter News Title", "")
text = st.text_area("📝 Enter News Content", "", height=250)

# Utility: Extract domain from title
def extract_domain(text):
    domain_patterns = ['cnn', 'fox', 'bbc', 'nbc', 'nyt', 'reuters', 'aljazeera', 'guardian', 'huffpost', 'abc']
    text = str(text).lower()
    for keyword in domain_patterns:
        if keyword in text:
            return keyword
    return 'other'

# Prediction
if st.button("🔍 Predict"):
    if not text.strip():
        st.warning("⚠️ Please enter article content to make a prediction.")
    else:
        domain = extract_domain(title)
        X_text = vectorizer.transform([text])
        X_domain = encoder.transform([[domain]])
        X_final = hstack([X_text, X_domain])
        
        prediction = model.predict(X_final)[0]
        proba = model.predict_proba(X_final)[0]

        label = "REAL" if prediction == 1 else "FAKE"
        confidence = round(max(proba) * 100, 2)

        st.success(f"🧠 Prediction: **{label}**")
        st.info(f"📊 Confidence: **{confidence}%**")

        # Add polarity info
        polarity = TextBlob(text).sentiment.polarity
        subjectivity = TextBlob(text).sentiment.subjectivity

        st.write(f"🧾 **Sentiment Polarity**: {polarity:.2f}")
        st.write(f"🧾 **Sentiment Subjectivity**: {subjectivity:.2f}")

        st.markdown("---")

# Comparison Plot
if st.checkbox("📊 Show Model Comparison Plot"):
    st.image("compare_plots/logistic_comparison.png", caption="Untuned vs Tuned Logistic Regression")

# Insights Section
if st.checkbox("📘 Show Insights & Recommendations"):
    st.markdown("""
### 🔍 Insights

- **High Accuracy**: The tuned logistic regression model provides strong performance.
- **Domain Helps**: Metadata from article title (e.g., 'cnn', 'fox') enhances prediction.
- **TF-IDF is Effective**: Word importance matters significantly in distinguishing real vs fake.

### ⚠️ Limitations

- No understanding of sarcasm or deep semantics.
- Limited domain detection logic.
- No time-based analysis or temporal awareness.

### 💡 Recommendations

- Add temporal features (e.g., publishing date).
- Use clickbait detection for headline-based indicators.
- Upgrade to BERT or transformer models for deep language understanding.
- Use explainable AI tools like SHAP or LIME for transparency.

### ⚖️ Ethics & Deployment Notes

- Avoid automatic censorship: Always verify with human reviewers.
- Beware of bias in training data.
- Continuously update the model as misinformation tactics evolve.
""")

# Footer
st.markdown("---")
st.markdown("🚨 **Disclaimer:** This tool is for educational purposes only. Predictions should not be used for real-world news judgment without human oversight.")

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from ftfy import fix_text

# Download NLTK data if not already
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load data
df = pd.read_csv("fake_or_real_news.csv")

# Drop unnamed column
df.drop(columns=[col for col in df.columns if "unnamed" in col.lower()], inplace=True)

# Initialize tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text cleaning and normalization
def clean_text(text):
    if pd.isnull(text):
        return ''
    # Fix encoding issues like â€™ to ’
    text = fix_text(text)
    # Lowercase
    text = text.lower()
    # Remove any garbled leftovers
    text = re.sub(r'[â€™“”‘’–—•›‹«»]', '', text)
    # Remove non-alphabetic characters (numbers, symbols)
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and lemmatize
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(cleaned)

# Apply cleaning to 'text' column
df['clean_text'] = df['text'].apply(clean_text)

# Word Count feature
df['word_count'] = df['clean_text'].apply(lambda x: len(x.split()))

# Sentiment polarity (TextBlob)
df['sentiment'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Term Frequency Matrix (CountVectorizer)
vectorizer = CountVectorizer(max_features=1000)
term_matrix = vectorizer.fit_transform(df['clean_text'])

# Create word cloud for FAKE news articles
fake_text = ' '.join(df[df['label'] == 'FAKE']['clean_text'])

wordcloud = WordCloud(width=800, height=400, background_color='white').generate(fake_text)

# Show word cloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud for FAKE News")
plt.show()

# Save enriched data (optional)
df.to_csv("cleaned_news_with_features.csv", index=False)

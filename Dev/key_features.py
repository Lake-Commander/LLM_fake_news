import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import nltk
import os

# Download necessary resources
nltk.download('punkt')

# Load cleaned dataset (explicit encoding to avoid Excel misreads)
df = pd.read_csv("cleaned_news_with_features.csv", encoding='utf-8')

# ========== Feature Engineering ==========

# Word count
df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

# Sentiment polarity and subjectivity
df['sentiment_polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['sentiment_subjectivity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# Save enriched dataset
df.to_csv("feature_enriched_news.csv", index=False)
print("✅ Enriched dataset saved as 'feature_enriched_news.csv'.")

# ========== Term Frequency ==========

def extract_term_frequencies(dataframe, column='text', max_features=100):
    print("🔍 Extracting term frequencies...")
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(dataframe[column])
    tf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    tf_df.to_csv("term_frequencies.csv", index=False)
    print("✅ Term frequencies saved as 'term_frequencies.csv'.")
    return tf_df

# Run term frequency extraction
extract_term_frequencies(df)

# ========== Word Cloud for FAKE News ==========

def generate_fake_news_wordcloud(dataframe, column='text'):
    print("☁️ Generating word cloud for FAKE news...")
    fake_text = " ".join(dataframe[dataframe['label'] == 'FAKE'][column].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords='english').generate(fake_text)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Word Cloud for FAKE News Articles")
    plt.tight_layout(pad=0)
    plt.savefig("fake_news_wordcloud.png")
    plt.close()
    print("✅ Word cloud saved as 'fake_news_wordcloud.png'.")

# Generate word cloud
generate_fake_news_wordcloud(df)

print("🎉 All tasks completed successfully.")

import pandas as pd
import re
from textblob import TextBlob

# Load the cleaned dataset
df = pd.read_csv("feature_enriched_news.csv")

# ========== Metadata-Based Feature Engineering ==========

# Title length in words
df['title_word_count'] = df['title'].astype(str).apply(lambda x: len(x.split()))

# Count of all caps words in title (e.g., "BREAKING")
df['title_all_caps_count'] = df['title'].astype(str).apply(lambda x: len([w for w in x.split() if w.isupper()]))

# Count of numerical digits in title
df['title_number_count'] = df['title'].astype(str).apply(lambda x: len(re.findall(r'\d+', x)))

# Count of exclamation marks in title
df['title_exclam_count'] = df['title'].astype(str).apply(lambda x: x.count('!'))

# Count of question marks in title
df['title_question_count'] = df['title'].astype(str).apply(lambda x: x.count('?'))

# Clickbait-like pattern detection
def detect_clickbait(title):
    clickbait_phrases = [
        r"^you won't believe", r"^this is what happens", r"^what happened next",
        r"^shocking", r"^this will blow your mind", r"^can't believe"
    ]
    title_lower = str(title).lower()
    return int(any(re.search(p, title_lower) for p in clickbait_phrases))

df['clickbait_flag'] = df['title'].astype(str).apply(detect_clickbait)

# Sentiment subjectivity of title
df['title_subjectivity'] = df['title'].astype(str).apply(lambda x: TextBlob(x).sentiment.subjectivity)

# Save enriched dataset
df.to_csv("fake_or_real_news_with_metadata.csv", index=False)
print("✅ Metadata-enriched dataset saved as 'fake_or_real_news_with_metadata.csv'.")

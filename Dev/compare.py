import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.sparse import hstack
import matplotlib.pyplot as plt

# === Step 1: Load data ===
df = pd.read_csv("transformed.csv")
X_text = df['text']
X_domain = df['domain']
y = df['label']

X_text_train, X_text_test, X_domain_train, X_domain_test, y_train, y_test = train_test_split(
    X_text, X_domain, y, test_size=0.2, random_state=42
)

# === Step 2: Load vectorizer and encoder ===
tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
domain_encoder = joblib.load('models/domain_encoder.pkl')

X_text_test_tfidf = tfidf_vectorizer.transform(X_text_test)
X_domain_test_encoded = domain_encoder.transform(X_domain_test.values.reshape(-1, 1))
X_test_combined = hstack([X_text_test_tfidf, X_domain_test_encoded])

# === Step 3: Load models ===
untuned_model = joblib.load('models/logistic_model.pkl')
tuned_model = joblib.load('models/tuned_logreg.pkl')

# === Step 4: Evaluation function ===
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, pos_label='FAKE'),
        'Recall': recall_score(y_true, y_pred, pos_label='FAKE'),
        'F1 Score': f1_score(y_true, y_pred, pos_label='FAKE')
    }

# === Step 5: Compare results ===
results = {
    'Untuned Logistic': evaluate_model(untuned_model, X_test_combined, y_test),
    'Tuned Logistic': evaluate_model(tuned_model, X_test_combined, y_test)
}

results_df = pd.DataFrame(results).T.round(4)
print("\nModel Comparison Table:\n")
print(results_df)

# === Step 6: Plot ===
plt.figure(figsize=(10, 6))
results_df.plot(kind='bar', figsize=(10, 6), colormap='Set2', edgecolor='black')
plt.title("Model Comparison: Untuned vs Tuned Logistic Regression", fontsize=14)
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1.05)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title="Metrics")
plt.tight_layout()

# === Step 7: Save plot ===
os.makedirs("compare_plots", exist_ok=True)
plot_path = "compare_plots/logistic_comparison.png"
plt.savefig(plot_path)
print(f"\n📊 Plot saved to: {plot_path}")

# Optional: show plot
# plt.show()

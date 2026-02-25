##  Disclaimer
> This application is strictly for educational and research purposes. It should not be used in any real-world or production setting to determine the authenticity of news without further validation and domain expertise.

# Fake News Detection with LLM Features

This repository contains an end-to-end machine learning pipeline for detecting fake news using traditional NLP features, metadata, and large language model (LLM)-derived features. It includes preprocessing, feature engineering, modeling and a deployable Streamlit web app.

---

## Project Overview

The goal is to classify news articles as **fake** or **real** using both text-based and metadata-based features. We use a combination of:

- Word count and text length statistics
- Sentiment features
- Domain and time-based features
- Term frequency metrics
- Features derived from large language models (LLMs)

---

##  Features

- Data preprocessing and cleaning
- Feature engineering (NLP + metadata)
- Correlation and bivariate analysis
- Model training and evaluation (Logistic Regression, Random Forest, etc.)
  Web app for predictions using Streamlit

---

##  Technologies

- Python 3.13
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- Joblib
- Streamlit

---

##  Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/Lake-Commander/LLM_fake_news.git
cd LLM_fake_news
```

###2. Install dependencies
Make sure you have Python 3.13 installed.
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

## Project Structure

```
LLM_fake_news/
├── data/               # Raw and processed datasets
├── models/             # Trained models and scalers
├── output_graphs/      # Visualizations and EDA outputs
│   ├── correlation/    # Correlation heatmaps
│   └── bivariate/      # Bivariate plots (boxplots, scatterplots, etc.)
├── app.py              # Streamlit web app
├── requirements.txt    # Python dependencies
├── fake_or_real.ipynb  # Guide to retrain with your data
└── README.md           # Project documentation
```


##  Train Your Own Model
**To train a new model with your dataset:**

1. Prepare your dataset in CSV format. Use the sample datasets here for schema reference.

2. Open and run fake_or_real.ipynb in Jupyter Notebook.

3. Follow the steps to preprocess, train, and save the model.

4. Ensure your app loads the new model path.

##  Deployed App
Access the live app:
 [Click here to open the app](https://llm-fake-or-real-news.streamlit.app/).

##  Acknowledgments
This project was built under the guidance and mentorship of the 3MTT (Three Million Technical Talent) program by the National Information Technology Development Agency (NITDA), Nigeria.

We sincerely appreciate NITDA and the Federal Ministry of Communications, Innovation and Digital Economy for the opportunity to learn, grow, and contribute to Nigeria’s digital transformation journey.

Thank you for empowering Nigerian youths with the skills to build real-world solutions.




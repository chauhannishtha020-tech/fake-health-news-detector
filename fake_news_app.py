import streamlit as st
import joblib
import re
import numpy as np

# Load model + vectorizer (trained earlier)
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Common suspicious keywords in fake health news
suspicious_words = [
    "miracle", "cure", "instant", "guaranteed", "secret",
    "ancient", "shocking", "never before", "risk-free", "superfood"
]

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # remove punctuation/numbers
    return text

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]

    # Simple heuristic check with suspicious words
    keyword_flag = any(word in cleaned for word in suspicious_words)

    return prediction, keyword_flag

# Streamlit UI
st.title("📰 Fake Health News Detector")
st.write("Paste any health-related news/article below to check if it’s potentially **FAKE or REAL**.")

user_input = st.text_area("Enter health news text:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first!")
    else:
        result, keyword_flag = predict_news(user_input)

        if result == 1:
            st.error("🚨 This news is likely **FAKE**.")
        else:
            st.success("✅ This news seems **REAL**.")

        if keyword_flag:
            st.info("🔎 Suspicious keywords detected (e.g., miracle, cure, instant). Be cautious!")

import streamlit as st
import joblib
import re
import numpy as np

# Load model + vectorizer (you already trained this earlier)
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Common suspicious keywords in fake health news
suspicious_words = [
    "miracle", "cure", "instant", "guaranteed", "secret",
    "ancient", "shocking", "never before", "risk-free", "superfood"
]

def highlight_keywords(text):
    highlighted = text
    for word in suspicious_words:
        pattern = re.compile(rf"({word})", re.IGNORECASE)
        highlighted = pattern.sub(r"**\1**", highlighted)
    return highlighted

# Streamlit UI
st.title("📰 Fake Health News Detector")
st.write("Paste a news headline or short article below to check reliability.")

user_input = st.text_area("Enter Health News Text")

if st.button("Check News"):
    if user_input.strip():
        # Vectorize input
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # Result
        label = "✅ Real / Reliable" if prediction == 1 else "❌ Fake / Misleading"
        confidence = round(np.max(proba) * 100, 2)

        st.subheader("Result")
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence Score: **{confidence}%**")

        # Highlight suspicious words
        st.subheader("Suspicious Keywords Found")
        highlighted_text = highlight_keywords(user_input)
        st.markdown(highlighted_text)

        # Suggestion
        if prediction == 0:
            st.warning("⚠️ This might be misleading. Consider checking WHO or CDC websites for reliable info.")
    else:
        st.error("Please enter some text to analyze.")

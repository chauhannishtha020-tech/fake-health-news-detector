import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake Health News Detector")
st.write("Enter a news headline or article text to check if it's **Real or Fake**.")

# User input
user_input = st.text_area("Paste your news text here:")

if st.button("Check News"):
    if user_input.strip() != "":
        # Preprocess and predict
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        # Show result
        if prediction == 1:
            st.success("✅ This news seems **REAL**.")
        else:
            st.error("❌ This news seems **FAKE**.")

        # Show probabilities
        st.subheader("Prediction Confidence")
        st.write(f"Real: {proba[1]*100:.2f}%")
        st.write(f"Fake: {proba[0]*100:.2f}%")

        # Pie chart
        fig, ax = plt.subplots()
        labels = ["Fake", "Real"]
        ax.pie(proba, labels=labels, autopct="%1.1f%%", startangle=90,
               colors=["red", "green"], explode=(0.05, 0.05))
        ax.axis("equal")
        st.pyplot(fig)

        # Google search link
        st.subheader("🔎 Fact-check this news")
        search_url = f"https://www.google.com/search?q={user_input}"
        st.markdown(f"[Search this news on Google]({search_url})")
    else:
        st.warning("⚠️ Please enter some news text first.")

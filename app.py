import streamlit as st
import random
# Fake Health News Detector - Simple Prototype

st.title("🩺 Fake Health News Detector")

st.write("Enter a health news headline or statement below, and the model will predict if it's likely **real or fake**.")

# User input
headline = st.text_area("Enter Health News:")

if st.button("Check"):
    if headline.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Dummy prediction (for prototype)
        result = random.choice(["✅ Real News", "❌ Fake News"])
        st.success(f"Prediction: {result}")
      

import streamlit as st
import joblib

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.title("📰 Fake News Detector")
st.markdown("Enter a news headline or content and I’ll predict if it's fake or real.")

user_input = st.text_area("Paste the news text here:", height=200)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        X_input = vectorizer.transform([user_input])
        prediction = model.predict(X_input)[0]
        if prediction == 1:
            st.success("✅ This news appears to be Real.")
        else:
            st.error("🚫 This news appears to be Fake.")

import streamlit as st
import joblib
import re


# -----------------------------
# Load saved model and vectorizer
# -----------------------------

model = joblib.load("fake_news_svm.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# -----------------------------
# Text cleaning function
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)


# -----------------------------
# Title
# -----------------------------

st.title("📰 Fake News Detection")
st.write(
    "Enter a news headline or article and the machine learning model "
    "will classify it based on patterns learned from the training dataset."
)


# -----------------------------
# Input
# -----------------------------

article = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste a news headline or article here..."
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Predict"):

    if article.strip() == "":
        st.warning("Please enter a news article or headline.")

    else:

        cleaned_article = clean_text(article)

        article_features = tfidf.transform([cleaned_article])

        prediction = model.predict(article_features)[0]

        probabilities = model.predict_proba(article_features)[0]

        fake_probability = float(probabilities[0] * 100)
        real_probability = float(probabilities[1] * 100)


        # -----------------------------
        # Display prediction
        # -----------------------------

        st.subheader("Prediction")

        if prediction == 1:
            st.success("✅ REAL")
        else:
            st.error("⚠️ POTENTIALLY MISLEADING")


        # -----------------------------
        # Display probabilities
        # -----------------------------

        st.subheader("Model Probabilities")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Potentially Misleading",
                f"{fake_probability:.2f}%"
            )

        with col2:
            st.metric(
                "Real",
                f"{real_probability:.2f}%"
            )


        # -----------------------------
        # Probability chart
        # -----------------------------

        st.subheader("Prediction Probability")

        st.progress(
            int(real_probability),
            text=f"Real: {real_probability:.2f}%"
        )

        st.progress(
            int(fake_probability),
            text=f"Potentially Misleading: {fake_probability:.2f}%"
        )


# -----------------------------
# Disclaimer
# -----------------------------

st.divider()

st.caption(
    "Note: This model identifies patterns learned from the training "
    "dataset. It does not independently verify the factual accuracy "
    "of news claims."
)

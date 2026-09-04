import streamlit as st
import pickle


# Page settings
st.set_page_config(
    page_title="Movie Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# Load the trained model and TF-IDF vectorizer
with open("sentiment_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Custom CSS
st.markdown("""
<style>

body {
    background-color: #f5f7fb;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 25px;
}

.footer {
    text-align: center;
    color: #888;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# Header
st.markdown(
    '<div class="title">🎬 Movie Sentiment Analyzer</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find out whether a movie review is positive or negative</div>',
    unsafe_allow_html=True
)

st.divider()


# Review input
st.subheader("📝 Enter Your Movie Review")

review = st.text_area(
    "Write your review:",
    placeholder="Example: The movie was amazing and the acting was excellent!",
    height=180
)


# Analyze review
if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not review.strip():
        st.warning("Please enter a movie review first.")

    else:
        # Convert the review into TF-IDF features
        review_data = vectorizer.transform([review])

        # Predict sentiment
        prediction = model.predict(review_data)[0]

        # Get prediction confidence
        probabilities = model.predict_proba(review_data)[0]
        confidence = max(probabilities) * 100

        st.divider()
        st.subheader("📊 Result")

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.progress(int(confidence))


# Example reviews
st.divider()

st.subheader("💡 Try These Examples")

col1, col2 = st.columns(2)

with col1:
    st.write("😊 **Positive**")
    st.write(
        "This movie was fantastic! "
        "The acting and story were excellent."
    )

with col2:
    st.write("😞 **Negative**")
    st.write(
        "This movie was boring and disappointing. "
        "The story was terrible."
    )


# About the project
st.divider()

with st.expander("ℹ️ About This Project"):

    st.write("""
    ### Movie Sentiment Analysis

    This project uses Natural Language Processing (NLP)
    and Machine Learning to analyze movie reviews.

    The system predicts whether a review is:

    - 😊 Positive
    - 😞 Negative

    **Model:** Logistic Regression

    **Text Processing:** TF-IDF

    **Learning:** Supervised Machine Learning

    **Dataset:** IMDB Movie Reviews Dataset

    **Technology:** Python, Scikit-learn and Streamlit
    """)


# Footer
st.markdown(
    '<div class="footer">Built with Python, NLP & Machine Learning 🚀</div>',
    unsafe_allow_html=True
)
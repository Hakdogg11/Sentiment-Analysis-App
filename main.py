import streamlit as st
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import cleantext

# Function to clean text data
def clean_text(text):
    cleaned_text = cleantext.clean(str(text), clean_all=False, extra_spaces=True,
                        stopwords=True, lowercase=True, numbers=True, punct=True)
    return cleaned_text

# Function to correct words in the text using TextBlob
def correct_text(text):
    blob = TextBlob(text)
    corrected_text = str(blob.correct())
    return corrected_text

# Function to analyze sentiment using VADER
def analyze_sentiment_VADER(text):
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(text)
    compound_score = vs['compound']

    # Determine sentiment prediction label based on compound score
    if compound_score >= 0.05:
        sentiment_prediction = "Positive 😀"
        sentiment_color = "green"
    elif compound_score <= -0.05:
        sentiment_prediction = "Negative 😞"
        sentiment_color = "red"
    else:
        sentiment_prediction = "Neutral 😐"
        sentiment_color = "yellow"

    return sentiment_prediction, sentiment_color

# Streamlit App Title
st.title('Twitter Sentiment Analyzer')

# Text Input Section
st.header('Analyze Text')

# Text area for user input
text_input = st.text_area('Enter Text:', height=200, key="text_input")

# Checkbox to enable word correction
enable_correction = st.checkbox('Enable Word Correction')

# Button to trigger sentiment analysis for text input
analyze_button = st.button('Analyze Sentiment', key="analyze_text_button")

# Display sentiment prediction for text input
if analyze_button and text_input:
    cleaned_text = clean_text(text_input)
    if enable_correction:
        corrected_text = correct_text(cleaned_text)
        st.text_area('Corrected Text:', value=corrected_text, height=200)
        cleaned_text = corrected_text
    sentiment_prediction, sentiment_color = analyze_sentiment_VADER(cleaned_text)
    st.markdown(f"<p style='color:{sentiment_color}; font-size: 20px;'>Sentiment Prediction: {sentiment_prediction}</p>", unsafe_allow_html=True)

# CSV Upload Section
st.header('Analyze CSV')

# File uploader for CSV files
uploaded_file = st.file_uploader('Upload CSV File:', type=['csv'])

# Button to trigger CSV file analysis
analyze_csv_button = st.button('Analyze Sentiment', key="analyze_csv_button")

# Display loading spinner while performing CSV analysis
if analyze_csv_button and uploaded_file is not None:
    with st.spinner('Analyzing sentiment for CSV...'):
        try:
            df = pd.read_csv(uploaded_file)
            df['cleaned_tweet'] = df['tweet'].apply(clean_text)
            if enable_correction:
                df['cleaned_tweet'] = df['cleaned_tweet'].apply(correct_text)
            df['sentiment_prediction'], df['sentiment_color'] = zip(*df['cleaned_tweet'].apply(analyze_sentiment_VADER))
            st.write(df.head(10))

            # Download button to download predicted data as CSV
            csv_data = df.drop(columns=['sentiment_color']).to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predicted Data as CSV",
                data=csv_data,
                file_name='predicted_sentiment.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error: {e}")

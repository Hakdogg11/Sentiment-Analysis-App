import streamlit as st
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import cleantext

vader_analyzer = SentimentIntensityAnalyzer()

def vader_sentiment_label(score):
    if score >= 0.05:
        return 'Positive 😀'
    elif score <= -0.05:
        return 'Negative 😞'
    else:
        return 'Neutral 😐'

def clean_text(text):
    if text is None or text.strip() == "":
        return ""
    cleaned_text = cleantext.clean(str(text), clean_all=False, extra_spaces=True,
                                stopwords=True, lowercase=True, numbers=True, punct=True)
    return cleaned_text if cleaned_text is not None else ""

def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sentiment_scores = vader_analyzer.polarity_scores(cleaned_text)
    compound_score = sentiment_scores['compound']
    sentiment_label = vader_sentiment_label(compound_score)
    return sentiment_label

def set_bg_image():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://2.bp.blogspot.com/-KuSMlNbTMOU/UN7IJO0EtLI/AAAAAAABEo0/XhUx4uYN0AU/s1600/twitter_background_by_ormanclark-d308zg6.jpg");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_image()

col1, col2 = st.columns([1, 9])

with col1:
    st.image('Twitter-Logo.png', width=100)  

with col2:
    st.markdown("<h1 style='color:white;'>Twitter Sentiment Analyzer</h1>", unsafe_allow_html=True)

st.markdown("<h1 style='color:white;'>Analyze Text</h1>", unsafe_allow_html=True)

text_input = st.text_area('Enter Text', height=200, key="text_input")

analyze_button = st.button('Analyze Sentiment', key="analyze_text_button")

if analyze_button and text_input:
    sentiment_prediction = predict_sentiment(text_input)
    st.write(f"Sentiment Prediction: {sentiment_prediction}")

st.markdown("<h1 style='color:white;'>Analyze CSV</h1>", unsafe_allow_html=True)

column_name = st.text_input('Enter the column name that contains the text data:', 'tweet')

uploaded_file = st.file_uploader('Upload CSV File:', type=['csv'])

analyze_csv_button = st.button('Analyze Sentiment', key="analyze_csv_button")

if analyze_csv_button and uploaded_file is not None:
    with st.spinner('Analyzing sentiment for CSV...'):
        try:
            df = pd.read_csv(uploaded_file)
            if column_name not in df.columns:
                st.error(f"The specified column name '{column_name}' does not exist in the CSV file.")
                st.stop()
            df[column_name].fillna('', inplace=True)
            df['cleaned_text'] = df[column_name].apply(clean_text)
            df['sentiment_prediction'] = df['cleaned_text'].apply(predict_sentiment)
            st.write(df.head(10))

            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predicted Data as CSV",
                data=csv_data,
                file_name='predicted_sentiment.csv',
                mime='text/csv'
            )
        except Exception as e:
            st.error(f"Error: {e}")

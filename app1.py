# Import necessary libraries
import streamlit as st
import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
from pydub import AudioSegment
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Title of the Dashboard
st.title("Customer Call Data Analysis Dashboard")

# Sidebar for uploading files
st.sidebar.title("Upload Section")
audio_file = st.sidebar.file_uploader("Upload a customer call audio file (.wav)", type=["wav"])
csv_file = st.sidebar.file_uploader("Upload a customer call transcription CSV file", type=["csv"])

# Task 1 - Speech to Text and Audio File Info
if audio_file:
    st.header("Task 1 - Speech to Text and Audio Information")

    recognizer = sr.Recognizer()
    
    # Convert the uploaded audio file to text
    audio_segment = AudioSegment.from_file(audio_file)
    number_channels = audio_segment.channels
    frame_rate = audio_segment.frame_rate

    st.write(f"**Number of Channels**: {number_channels}")
    st.write(f"**Frame Rate**: {frame_rate}")
    
    # Speech Recognition
    audio_data = sr.AudioFile(audio_file)
    with audio_data as source:
        audio = recognizer.record(source)
    transcribed_text = recognizer.recognize_google(audio)
    
    st.write("**Transcribed Text**: ")
    st.write(transcribed_text)

# Task 2 - Sentiment Analysis
if csv_file:
    st.header("Task 2 - Sentiment Analysis")
    
    # Load customer call transcriptions CSV file
    df = pd.read_csv(csv_file)
    
    # Perform Sentiment Analysis using Vader
    sid = SentimentIntensityAnalyzer()

    def find_sentiment(text):
        scores = sid.polarity_scores(text)
        compound_score = scores['compound']
        if compound_score >= 0.05:
            return 'positive'
        elif compound_score <= -0.05:
            return 'negative'
        else:
            return 'neutral'

    df['sentiment_predicted'] = df.apply(lambda row: find_sentiment(row["text"]), axis=1)

    # Display the dataframe with sentiment prediction
    st.write(df[['text', 'sentiment_predicted']])

    # True positives for positive label
    true_positive = len(df.loc[(df['sentiment_predicted'] == df['sentiment_label']) &
                               (df['sentiment_label'] == 'positive')])
    st.write(f"**True Positives**: {true_positive}")

# Task 3 - Named Entity Recognition (NER)
    st.header("Task 3 - Named Entity Recognition (NER)")
    
    def extract_entities(text):
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents]
        return entities

    # Apply NER
    df['named_entities'] = df['text'].apply(extract_entities)
    st.write(df[['text', 'named_entities']])

    # Display the most frequent entity
    all_entities = [ent for entities in df['named_entities'] for ent in entities]
    entities_df = pd.DataFrame(all_entities, columns=['entity'])
    entities_counts = entities_df['entity'].value_counts().reset_index()
    entities_counts.columns = ['entity', 'count']
    
    most_freq_ent = entities_counts["entity"].iloc[0]
    st.write(f"**Most Frequent Entity**: {most_freq_ent}")

# Task 4 - Find Most Similar Text to Query
    st.header("Task 4 - Find Most Similar Text")
    
    # User input for query
    input_query = st.text_input("Enter a query to find similar customer calls:", "wrong package delivery")

    # Process input query
    processed_query = nlp(input_query)
    df['processed_text'] = df['text'].apply(lambda text: nlp(text))

    # Calculate similarity scores
    df['similarity'] = df['processed_text'].apply(lambda text: processed_query.similarity(text))
    df = df.sort_values(by='similarity', ascending=False)

    # Display the most similar text
    most_similar_text = df["text"].iloc[0]
    st.write(f"**Most Similar Text**: {most_similar_text}")
    
    # Display top 5 similar texts
    st.write("**Top 5 Similar Texts**:")
    st.write(df[['text', 'similarity']].head(5))

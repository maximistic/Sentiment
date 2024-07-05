#dependencies and libraries
!pip install vaderSentiment
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

#Pre-processing
def preprocess_data(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.date
    df['content'] = df['content'].fillna('')
    return df

news_df = preprocess_data(news_df)

# Feature Engineering
# Normalize the sentiment scores to a scale of -5 to 5
def scale_sentiment(score):
    return score * 5

sentiment_by_date['scaled_sentiment'] = sentiment_by_date['sentiment'].apply(scale_sentiment)

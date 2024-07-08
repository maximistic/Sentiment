!pip install vaderSentiment
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Step 1: Data Collection
news_api_key = 'api_key'
ticker = 'NVDA'
url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={news_api_key}'

response = requests.get(url)
if response.status_code == 200:
    news_data = response.json()['articles']
    news_df = pd.DataFrame(news_data)
    news_df.to_csv(f'{ticker}_news.csv', index=False)
    print(f"News articles for {ticker} saved to {ticker}_news.csv")
else:
    print(f"Error: {response.status_code}")
    print("Response:", response.json())

# Pre-processing
def preprocess_data(df):
    df['publishedAt'] = pd.to_datetime(df['publishedAt']).dt.date
    df['content'] = df['content'].fillna('')
    return df

news_df = preprocess_data(news_df)

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    score = analyzer.polarity_scores(text)
    return score['compound']

news_df['sentiment'] = news_df['content'].apply(analyze_sentiment)

# Aggregate sentiment scores by date
sentiment_by_date = news_df.groupby('publishedAt').agg({'sentiment': 'mean'}).reset_index()

# Feature Engineering
# Normalize the sentiment scores to a scale of -5 to 5
def scale_sentiment(score):
    return score * 5

sentiment_by_date['scaled_sentiment'] = sentiment_by_date['sentiment'].apply(scale_sentiment)

# Model Training
# Splitting the data
sentiment_by_date['date_numeric'] = sentiment_by_date['publishedAt'].apply(lambda date: date.toordinal())
X = sentiment_by_date[['date_numeric']]
y = sentiment_by_date['scaled_sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)

# Calculate MSE
mse = np.mean((y_pred - y_test) ** 2)
print(f"Mean Squared Error: {mse}")

# Plotting
plt.figure(figsize=(14, 7))
plt.scatter(X_test['date_numeric'], y_test, color='blue', label='Actual Sentiment')
plt.scatter(X_test['date_numeric'], y_pred, color='red', label='Predicted Sentiment')
plt.xlabel('Date')
plt.ylabel('Sentiment Score')
plt.legend()
plt.show()

# Function to predict sentiment for a given date
def predict_sentiment(date, model):
    date_numeric = date.toordinal()
    prediction = model.predict([[date_numeric]])
    return prediction[0]

date = datetime(2023, 12, 31).date()
predicted_sentiment = predict_sentiment(date, model, sentiment_by_date)
print(f"Predicted Sentiment for {date}: {predicted_sentiment}")

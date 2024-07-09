import requests
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# Step 1: Data Collection
news_api_key = 'a3d5dce2b6f44a649088f97e4cf64b65'
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
sentiment_by_date['date_numeric'] = sentiment_by_date['publishedAt'].apply(lambda date: date.toordinal())

# Splitting the data
X = sentiment_by_date[['date_numeric']]
y = sentiment_by_date['scaled_sentiment']

# Define the model and parameter grid
model = RandomForestRegressor()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform K-Fold Cross-Validation with Hyperparameter Tuning
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold, scoring='neg_mean_squared_error', n_jobs=-1)

grid_search.fit(X, y)

# Best model from Grid Search
best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
y_pred = best_model.predict(X)

# Calculate MSE
mse = np.mean((y_pred - y) ** 2)
print(f"Mean Squared Error: {mse}")

# Plotting
plt.figure(figsize=(14, 7))
plt.scatter(X['date_numeric'], y, color='blue', label='Actual Sentiment')
plt.scatter(X['date_numeric'], y_pred, color='red', label='Predicted Sentiment')
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
predicted_sentiment = predict_sentiment(date, best_model)
print(f"Predicted Sentiment for {date}: {predicted_sentiment}")

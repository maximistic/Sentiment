import requests
import pandas as pd

# News API key (you need to sign up for a free API key at newsapi.org)
news_api_key = 'yourAPIKey'
ticker = 'TSLA'
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

import pandas as pd


data = pd.read_excel('tweets.xlsx')
tweets = pd.DataFrame(data)

import pandas as pd

def calculate_overall_negative_percentage(tweets):
    negative_tweets = tweets[tweets['labels_sentiment_0_sentiment'] == 'negative']
    negative_percentage = len(negative_tweets) / len(tweets) * 100
    return negative_percentage

# Calculate KPIs
overall_negative_percentage = calculate_overall_negative_percentage(tweets)
print(f"Overall Negative Sentiment Percentage: {overall_negative_percentage:.2f}%")



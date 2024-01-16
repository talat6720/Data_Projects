import pandas as pd

def calculate_sentiment_by_time(tweets, date_time_column='source_created_at', sentiment_column='labels_sentiment_0_sentiment'):
    # Ensure the 'source_created_at' column is in datetime format
    if not pd.api.types.is_datetime64_any_dtype(tweets[date_time_column]):
        tweets['hour'] = pd.to_datetime(tweets[date_time_column], format='%d/%m/%Y %H:%M:%S').dt.hour
    else:
        tweets['hour'] = tweets[date_time_column].dt.hour

    # Calculate and return the sentiment distribution by hour
    sentiment_by_hour = tweets.groupby('hour')[sentiment_column].value_counts(normalize=True)
    
    # Get the unique sentiment values in the DataFrame
    unique_sentiments = tweets[sentiment_column].unique()
    
    # Calculate the top 6 hours with the highest sentiment percentages for each sentiment
    top_hours_by_sentiment = {}
    for sentiment in unique_sentiments:
        top_hours = sentiment_by_hour.xs(sentiment, level=1).groupby(level='hour').max().nlargest(6)
        top_hours_by_sentiment[sentiment] = top_hours
    
    # Print the top 6 hours with the highest sentiment percentages for each sentiment
    for sentiment, top_hours in top_hours_by_sentiment.items():
        print(f"Top 6 hours with highest {sentiment} sentiment:")
        print(top_hours)
    
    return sentiment_by_hour

# Example usage:
tweets = pd.read_excel('tweets.xlsx') # Assuming the DataFrame is already loaded
sentiment_by_time = calculate_sentiment_by_time(tweets)

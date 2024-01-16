import pandas as pd

# Load Excel file into a pandas DataFrame
tweets = pd.read_excel('tweets.xlsx')

# Define your facility terms - assuming these are the topics you want to check within the 'topic' column
facility_terms = [
    'air conditioning', 'announcements', 'brakes', 'COVID', 'delays', 'doors', 'floor', 'handrails', 'hvac',
    'noise', 'plugs', 'roof', 'seats', 'service', 'station', 'tables', 'tickets/seat reservations', 'toilets', 
    'train general', 'vandalism', 'wifi', 'windows'
]

# Map sentiment labels to numerical scores
sentiment_mapping = {
    'negative': -1,
    'neutral': 0,
    'positive': 1
}

# Function to calculate the average sentiment scores for each facility term
def calculate_average_sentiment(tweets, sentiment_mapping):
    # Map the sentiment to scores
    tweets['sentiment_score'] = tweets['labels_sentiment_0_sentiment'].map(sentiment_mapping)
    # Group by 'topic' and calculate the mean sentiment score
    return tweets.groupby('labels_topic_0_topic')['sentiment_score'].mean()

# Function to rank facility terms by the number of negative tweets
def rank_facility_terms(tweets):
    # Filter for negative tweets
    negative_tweets = tweets[tweets['labels_sentiment_0_sentiment'] == 'negative']
    # Count the frequency of each topic
    return negative_tweets['labels_topic_0_topic'].value_counts()

# Calculate average sentiment scores for each facility term
average_sentiment = calculate_average_sentiment(tweets, sentiment_mapping)
print("Average Sentiment Scores for Each Facility Term:")
print(average_sentiment.sort_values(ascending=False))

# Rank the facility terms by the number of negative tweets
ranked_facility_terms = rank_facility_terms(tweets)
print("\nFacility Terms Ranked by Number of Negative Tweets:")
print(ranked_facility_terms.head(5))  # Modify the number as needed to get the top N facility terms


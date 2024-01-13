from transformers import pipeline

# Load the ZeroShot classifier model
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the sentiment classes
sentiment_classes = ["positive", "negative", "neutral"]

# Function to predict sentiment for a single tweet
def predict_sentiment(tweet):
    print("predict_sentiment execution started")
    result = classifier(tweet, sentiment_classes)
    print("predict_sentiment execution ended")
    print(result)
    return result['labels'][0]

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Load the dataset
# Replace the path with the correct one, for example:
# '/mnt/data/tweets_with_formatted_dates.csv' if you're running this in the current environment
# Load the dataset
tweets_new_df = pd.read_csv('json_to_csv_saurabh.csv')

# Data Cleaning

# Make a copy of the entire DataFrame instead of selecting specific columns
tweets_new_sub_df = tweets_new_df.copy()

# Convert 'source_created_at' column to datetime format (if not already)
tweets_new_sub_df['source_created_at'] = pd.to_datetime(tweets_new_sub_df['source_created_at'], errors='coerce')

# Print number of rows in the DataFrame
print("Number of rows in the DataFrame:", len(tweets_new_sub_df))

# Feature Engineering
# Extract the hour from the 'source_created_at' column
tweets_new_sub_df['hour'] = tweets_new_sub_df['source_created_at'].dt.hour

# Function to remove Twitter handles from text
def remove_twitter_handles(text):
    # Define the regex pattern for Twitter handles: @ followed by one or more word characters
    pattern = r'@\w+'
    # Replace the handles with an empty string
    return re.sub(pattern, '', text)

# Apply the function to the 'text' column and add the result as a new column 'updated_text'
tweets_new_sub_df['updated_text'] = tweets_new_sub_df['text'].apply(remove_twitter_handles)

# Function to remove emojis and URLs from text and return counts
def clean_text(text):
    # Count and remove emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    emoji_count = len(emoji_pattern.findall(text))
    text = emoji_pattern.sub(r'', text)
    
    # Count and remove URLs
    url_pattern = re.compile(r'http\S+|www\S+')
    url_count = len(url_pattern.findall(text))
    text = url_pattern.sub(r'', text)
    
    return text, emoji_count, url_count

# Assuming 'updated_text' column exists, apply the cleaning function and extract counts
tweets_new_sub_df['updated_text'], tweets_new_sub_df['emoji_count'], tweets_new_sub_df['url_count'] = \
    zip(*tweets_new_sub_df['updated_text'].map(clean_text))

# Convert text to lowercase
tweets_new_sub_df['updated_text'] = tweets_new_sub_df['updated_text'].str.lower()

# Define a new function that will also remove '@' and other non-alphanumeric characters
def remove_special_characters(text):
    # Count special characters and punctuations before cleaning
    special_chars_before = len(re.findall(r'[^\w\s]', text))
    # Remove special characters and punctuations
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    # Count special characters and punctuations after cleaning
    special_chars_after = len(re.findall(r'[^\w\s]', cleaned_text))
    return cleaned_text, special_chars_before, special_chars_after

# Apply the new function to the updated_text column
tweets_new_sub_df['cleaned_text'], tweets_new_sub_df['punctuations_before'], tweets_new_sub_df['punctuations_after'] = \
    zip(*tweets_new_sub_df['updated_text'].map(remove_special_characters))

# Define a function to remove numbers from text and return the count of numbers removed
def remove_numbers(text):
    # Find all numbers in the text
    numbers_found = re.findall(r'\d+', text)
    # Remove numbers
    cleaned_text = re.sub(r'\d+', '', text)
    # Count the number of digits found
    number_count = sum(len(number) for number in numbers_found)
    return cleaned_text, number_count

# Apply the function to remove numbers from the 'cleaned_text' column and count them
tweets_new_sub_df['cleaned_text'], tweets_new_sub_df['numbers_count'] = \
    zip(*tweets_new_sub_df['cleaned_text'].map(remove_numbers))

# Calculate the total number of numbers removed from the entire dataset
total_numbers_removed = tweets_new_sub_df['numbers_count'].sum()
print(f"Total number of numbers removed: {total_numbers_removed}")

# Calculate the total number of emojis, URLs, and punctuations before and after cleaning
total_emojis_removed = tweets_new_sub_df['emoji_count'].sum()
total_urls_removed = tweets_new_sub_df['url_count'].sum()
total_punctuations_before = tweets_new_sub_df['punctuations_before'].sum()
total_punctuations_after = tweets_new_sub_df['punctuations_after'].sum()

# Print the total number of emojis, URLs, and punctuations before and after cleaning
print(f"Total number of emojis removed: {total_emojis_removed}")
print(f"Total number of URLs removed: {total_urls_removed}")
print(f"Total number of punctuations before cleaning: {total_punctuations_before}")
print(f"Total number of punctuations after cleaning: {total_punctuations_after}")

# Tokenization
#nltk.download('punkt')  # Make sure to download the 'punkt' tokenizer
tweets_new_sub_df['tokens'] = tweets_new_sub_df['cleaned_text'].apply(word_tokenize)
initial_tokens = sum(tweets_new_sub_df['tokens'].apply(len))
print(f"Total tokens after tokenization: {initial_tokens}")

# Stopword Removal
#nltk.download('stopwords')  # Make sure to download the 'stopwords' set
stop_words = set(stopwords.words('english'))
tweets_new_sub_df['filtered_tokens'] = tweets_new_sub_df['tokens'].apply(lambda x: [word for word in x if word.lower() not in stop_words])
tokens_after_stopword_removal = sum(tweets_new_sub_df['filtered_tokens'].apply(len))
print(f"\nTotal tokens after stopword removal: {tokens_after_stopword_removal}")

# Stemming
stemmer = PorterStemmer()
tweets_new_sub_df['stemmed_tokens'] = tweets_new_sub_df['filtered_tokens'].apply(lambda x: [stemmer.stem(word) for word in x])
unique_tokens_before_stemming = len(set([token for sublist in tweets_new_sub_df['filtered_tokens'] for token in sublist]))
unique_tokens_after_stemming = len(set([token for sublist in tweets_new_sub_df['stemmed_tokens'] for token in sublist]))
print(f"\nUnique tokens before stemming: {unique_tokens_before_stemming}")
print(f"Unique tokens after stemming: {unique_tokens_after_stemming}")

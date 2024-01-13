import matplotlib.pyplot as plt
import pandas as pd

# Load data 
data = pd.read_csv('tweets_with_formatted_dates.csv')

# Check dimensions
print(data.shape)

# Convert column to datetime
data['source_created_at'] = pd.to_datetime(data['source_created_at'])


# Plot the distribution of tweet lengths
plt.hist(data['text'].str.len())
plt.xlabel('Tweet length')
plt.ylabel('Number of tweets')
plt.title('Distribution of tweet lengths')
plt.show()


# Group by year instead of day 
sentiment_counts = data.groupby([data['source_created_at'].dt.year, 'labels_sentiment_0_sentiment'])['text'].count().unstack()

# Plot
ax = sentiment_counts.plot(kind='bar', stacked=True, title='Tweet Sentiment by Year')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Tweets')
ax.legend(loc='upper left')

# Add labels to the bars
for bar in ax.containers:
    for rect in bar.patches:
        ax.annotate(f'{rect.get_height():.0f}', xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()), ha='center', va='center', fontsize=8)

# Add a text box to the plot
ax.text(0.6, 0.5, f'Total tweets: {sentiment_counts.sum().sum():.0f}', ha='center', va='center', fontsize=16)

plt.show()


# Get the number of tweets in each sentiment category
counts = data['labels_sentiment_0_sentiment'].value_counts()

# Create a bar chart
plt.bar(counts.index, counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Number of Tweets')
plt.title('Tweet Sentiment Distribution')
plt.show()


from collections import Counter
import wordcloud

# Convert the 'labels_topic_0_topic' column to a list of strings
topics = data['labels_topic_0_topic'].tolist()

# Create a word cloud
wordcloud = wordcloud.WordCloud().generate(' '.join(topics))

# Plot the word cloud
plt.figure(figsize=(10, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# Get the number of tweets for each topic
topic_counts = data['labels_topic_0_topic'].value_counts()

# Create a bar chart
plt.bar(topic_counts.index, topic_counts.values)
plt.xlabel('Topic')
plt.ylabel('Number of Tweets')
plt.title('Topic Distribution')
plt.xticks(rotation=45)
plt.show()

# Get the number of unique topics
num_topics = data['labels_topic_0_topic'].nunique()

# Print the number of topics
print('Number of topics:', num_topics)

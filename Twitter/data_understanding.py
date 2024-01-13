import pandas as pd

# Load data 
df = pd.read_csv('tweets_with_formatted_dates.csv')

# Check dimensions
print(df.shape)

df.describe()

# Check column names
print(df.columns)

# Check null values
print(df.isnull().sum())

# Check duplicate rows
print(df.duplicated().sum()) 

# Remove duplicates if any
df = df.drop_duplicates()

# Check sample rows 
print(df.head())

# Check topic label distribution
print(df['labels_topic_0_topic'].value_counts())

# Check sentiment label distribution
print(df['labels_sentiment_0_sentiment'].value_counts())

# Map tweets to stations using latitude/longitude
stations = {
    (51.50853, -0.12574) : 'London Bridge',
    (53.38297, -1.4659) : 'York'
}

df['station'] = df.apply(lambda x: stations.get((x['latitude'], x['longitude']), 'Unknown'), axis=1)

print(df['station'].value_counts())
import pandas as pd
from textblob import TextBlob

# Load Excel file into a pandas DataFrame
data = pd.read_excel('tweets.xlsx')
tweets = pd.DataFrame(data) 

# =====================
# Maintainace KPIs
# =====================

# Broken facilities counter
facility_terms = ['air conditioning', 'announcements', 'brakes', 'COVID', 'delays', 'doors', 'floor', 'handrails', 'hvac',
    'noise', 'plugs', 'roof', 'seats', 'service', 'station', 'tables', 'tickets/seat reservations', 'toilets', 'train general', 'vandalism', 'wifi', 'windows'] 

def count_broken_facilities(text):
  count = 0
  for term in facility_terms:
    if term in text.lower():
      count += 1
  return count

tweets['broken_facilities'] = tweets['labels_topic_0_topic'].apply(count_broken_facilities)
print(tweets['broken_facilities'].sum())




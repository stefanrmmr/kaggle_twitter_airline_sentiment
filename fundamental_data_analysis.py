# Fundamental Data Analysis for Twitter tweet text sentiment analysis
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

import pandas as pd
from plotting_framework import *

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory


print("\n_______FUNDAMENTAL DATA ANALYSIS________\n")

# IMPORT DATA TWEETS: Airlines
df_tweets_air_full = pd.read_csv('tweets_data/Tweets_airlines.csv')
df_selected = df_tweets_air_full[['airline_sentiment', 'tweet_location', 'airline']]

# abbreviate Airline names
df_selected['airline'].replace({"Virgin America": "Virgin"}, inplace=True)
df_selected['airline'].replace({"Southwest": "Southw."}, inplace=True)
df_selected['airline'].replace({"US Airways": "US Airw."}, inplace=True)

df_neg = df_selected[df_selected['airline_sentiment'] == 'negative']
df_ntr = df_selected[df_selected['airline_sentiment'] == 'neutral']
df_pos = df_selected[df_selected['airline_sentiment'] == 'positive']
print(df_selected.info(), "\n")

tweet_counts = df_selected['airline'].value_counts()
tweets_amount = len(df_selected)
print(f"TOTAL    tweet count: {tweets_amount}")
print(f"negative tweet count: {len(df_neg)}")
print(f"neutral  tweet count: {len(df_ntr)}")
print(f"positive tweet count: {len(df_pos)}")

print("\n", tweet_counts, "\n")

dict_air = []
airlines = list(df_selected['airline'].unique())

for airline in airlines:  # calculate distribution of tweet data among airlines
    tweet_count = list(df_selected['airline']).count(airline)
    tweet_percentage = round(tweet_count/tweets_amount, 4)
    count_neg = list(df_neg['airline']).count(airline)
    count_ntr = list(df_ntr['airline']).count(airline)
    count_pos = list(df_pos['airline']).count(airline)

    dict_air.append({'airline_name': airline,
                     'tweet_count': tweet_count,
                     'percentage': tweet_percentage,
                     'count_neg': count_neg,
                     'count_ntr': count_ntr,
                     'count_pos': count_pos})

# plot_tweet_distribution(dict_air)
plot_sentiment_distribution(dict_air)

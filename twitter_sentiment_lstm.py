#
#
# sentiment classification of US Airline tweets

import numpy as np
import pandas as pd
# for interaction with Twitter
import tweepy as tw
# for PREPROCESSING
import re    # RegEx for removing non-letter characters
import nltk  #natural language processing
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.preprocessing import OneHotEncoder

def get_text(twitter_data): # get DataFrame with tweet_id and associated text
    text_data = pd.DataFrame(columns = ["tweet_id", "text"])
    for index, id in enumerate(twitter_data.loc[:10, "tweet_id"]):
        try: # check if id is working
            text = api.get_status(id).text
            print(id)
        except: # if not set text to nan
            text = np.nan
            print(f"failed id:{id}")
        tweet_info = {"tweet_id" : f"{id}", "text" : text, "sentiment" : twitter_data.loc[index, "airline_sentiment"]}
        text_data = text_data.append(tweet_info, ignore_index = True)
    return text_data

def tweet_to_words(tweet):
    # cleaning of raw tweet
    text = tweet.lower()                            # lower case
    text = re.sub(r"http\S+", "", text)             # text remove hyperlinks
    text = re.sub(r"#", "", text)                   # text remove hashtag symbol
    text = re.sub(r"@\S+", "", text)                # text remove @mentions
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)       # remove non letters
    #text = re.sub(r"^RT[\s]+", "", text)            # remove retweet text "RT"
    # tokenization
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    #words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words

if __name__ == "__main__":
    # Twitter developer keys and setup
    ###
    consumer_key = "dBVOKYBYosXxTC9Z5B4BCilLm"
    consumer_key_secret = "gER2UBsOGJ8FtDGVfRPbXzuTskd9agbr3lNcuEPHT2iBYm641o"
    access_token = "1414641981570134020-UQC0Qk3fTowwDJLCqszONYnCSCwquj"
    access_token_secret = "XW3X5sWIx0V5RZ3FR2TA1XcP2AeqvhZCrdimPTcTusp6j"

    auth = tw.OAuthHandler(consumer_key, consumer_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth)
    ###

    # load tweets
    twitter_data = pd.read_csv("Tweets.csv") ###### SENTIMENT CONFIDENCE?
    print(twitter_data.head())
    # get DataFrame with tweet_id and text
    #text_data = get_text(twitter_data)
    text_data = twitter_data.loc[:, ["text", "airline_sentiment"]]
    print(text_data.head())
    print(text_data.isnull().sum())
    # drop rows/examples where either no text could be fetched or no sentiment was assigned
    text_data.dropna(axis = 0, how = "any", inplace = True)
    print(text_data.head())
    # label encode sentiments with direct mapping
    sent_map = {"negative" : -1.0, "neutral" : 0.0, "positive" : 1.0}
    text_data["sentiment"] = text_data["airline_sentiment"].map(sent_map)
    print(text_data.head())
    ###########################################
    # Visualization
    ###########################################
    # PREPROCESSING
    #print(text_data["text"][0])
    #print(tweet_to_words(text_data["text"][0]))
    text_data["words"] = list(map(tweet_to_words, text_data["text"]))
    print(text_data["words"].head())


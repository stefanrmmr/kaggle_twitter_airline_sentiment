# FOR TWITTER LOGIN
# Adrian Brünger, Stefan Rummer, TUM, summer 2021import numpy as np

# imports
import numpy as np
import pandas as pd
import tweepy as tw

def get_text(twitter_data, api): # get DataFrame with tweet_id and associated text
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

def twitter_login():
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
    return api
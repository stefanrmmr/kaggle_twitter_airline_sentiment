# TODO sentiment von airline? VS Flüge einer bestimmten Airline pro tag
#  (seit Corona, 250 most significant tweets on average), plot verlauf (pos-neg)
#####################################################################################################
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

# imports
# GENERAL
import pickle
import numpy as np
import pandas as pd
# for PREPROCESSING
from preprocessing import *
# LSTM MODEL
from LSTM_model import *
#for interaction with TWITTER
import tweepy as tw
from twitter_login import *

if __name__ == "__main__":

    print("\n_______DAML_Twitter_Sentiment________\n")

    # EVALUATION
    # import best LSTM model and respective tokenizer and padder
    lstm_model = load_model(r"model_data_keras_embedding\best_model_air_5kvocab.hdf5")
    with open(r'model_data_keras_embedding\tokenizer_air_5kvocab.pickle', 'rb') as handle_import:
        tokenizer_import = pickle.load(handle_import)  # load tokenizer
        # always use the same tokenizer so that word tokens are not changed
    with open(r'model_data_keras_embedding\padder_air_max_maxl.pickle', 'rb') as handle_import:
        max_length_import = pickle.load(handle_import)  # load tokenizer

    # Predict sentiments of a few example texts
    example_text = "The flight was perfect!" # TODO custom
    # "The flight was perfect!"
    # "This journey was a pleasure!"
    # "I disliked the service..."
    example_text = clean_text(example_text)
    print(example_text)
    print(predict_class(lstm_model, tokenizer_import, max_length_import, text = [example_text]))

    # SENTIMENT PREDICTION OF NEW AIRLINE TWEETS
    # Aggregate new data from Twitter
    # login twitter project
    api = twitter_login()

    search_words = "@United"
    date_since = "2019-12-01"
    date_until = "2021-09-05" #TODO with datetime

    tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since,
              until=date_until,
              result_type = "mixed",
              rpp = 100).items(250)

    twitter_data = pd.DataFrame(columns = ["date", "text"])
    for tweet in tweets:
        clean_tweet = clean_text(tweet.text)
        tweet_info = {"date" : tweet.created_at, "text" : clean_tweet}
        twitter_data = twitter_data.append(tweet_info, ignore_index = True)
    
    print(twitter_data.info())
    twitter_data.dropna(inplace = True)

    scores = predict_class(lstm_model, tokenizer_import, max_length_import, text = twitter_data["text"])
    print(scores)

    # TODO get most significant tweets and plot sentiment
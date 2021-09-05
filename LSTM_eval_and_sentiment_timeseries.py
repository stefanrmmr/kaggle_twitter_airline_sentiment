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
    lstm_model = load_model(r"model_data_keras_embedding\best_model_full_5kvocab.hdf5")
    with open(r'model_data_keras_embedding\tokenizer_full_5kvocab.pickle', 'rb') as handle_import:
        tokenizer_import = pickle.load(handle_import)  # load tokenizer
        # always use the same tokenizer so that word tokens are not changed
    with open(r'model_data_keras_embedding\padder_max_maxl.pickle', 'rb') as handle_import:
        max_length_import = pickle.load(handle_import)  # load tokenizer

    # Predict sentiments of a few example texts
    example_text = "Vaccination!" # TODO custom
    # "The flight was perfect!"
    # "This journey was a pleasure!"
    # "I disliked the service..."
    example_text = clean_text(example_text)
    print(example_text)
    predict_class(lstm_model, tokenizer_import, max_length_import, text = [example_text])

    # SENTIMENT PREDICTION OF NEW AIRLINE TWEETS
    # Aggregate new data from Twitter
    # login twitter project
    api = twitter_login()

    search_words = "lax"
    date_since = "2018-11-16"

    tweets = tw.Cursor(api.search,
              q=search_words,
              lang="en",
              since=date_since).items(5)

    tweet_list = []
    for tweet in tweets:
        clean_tweet = clean_text(tweet.text)
        print(tweet.text)
        #print(clean_tweet)
        tweet_list.append(clean_tweet)

    predict_class(lstm_model, tokenizer_import, max_length_import, text = tweet_list)

    # TODO get most significant tweets and plot sentiment
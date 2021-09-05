# PREPROCESSING
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

# This file contains 2 preprocessing functions to avoid redundant code

# imports
# GENERAL
import pickle
import pandas as pd
# for PREPROCESSING
import re    # RegEx for removing non-letter characters
import nltk  # natural language tool kit
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def prepare_dataframe(airline, filler_data, shuffle = True):
    # takes percentages of airline data and filler_data to be used as input
    # and returns concatenated DataFrame
    # load airline tweets
    df_tweets_air_full = pd.read_csv("tweets_data/Tweets_airlines.csv")
    #print(df_tweets_air_full.head())
    # get DataFrame with text and sentiment
    #text_data = get_text(twitter_data)
    df_tweets_air = df_tweets_air_full[["text", "airline_sentiment"]]
    print(f"Examples in airline data: {df_tweets_air.shape[0]}")
    df_tweets_air = df_tweets_air.rename(columns = {"text": "text", "airline_sentiment": "category"})
    df_tweets_air["category"] = df_tweets_air["category"].map({"negative": -1.0, "neutral": 0.0, "positive": 1.0})
    if shuffle == True:
        df_tweets_air = df_tweets_air.sample(frac=1)  # shuffle dataframe
    else: pass
    df_tweets_air = df_tweets_air.head(round(airline * df_tweets_air.shape[0])) # TODO custom
    print(f"USED examples of airline data: {df_tweets_air.shape[0]}\n")

    # load more tweets for more accurate results
    df_tweets_gen = pd.read_csv("tweets_data/Tweets_general.csv")
    #print(df_tweets_gen.head())
    df_tweets_gen = df_tweets_gen[["clean_text", "category"]]
    print(f"Examples in filler_data: {df_tweets_gen.shape[0]}")
    df_tweets_gen = df_tweets_gen.rename(columns = {"clean_text": "text", "category": "category"})
    if shuffle == True:
        df_tweets_gen = df_tweets_gen.sample(frac=1)  # shuffle dataframe
    else: pass
    df_tweets_gen = df_tweets_gen.head(round(filler_data * df_tweets_gen.shape[0]))  # TODO custom
    print(f"USED examples of filler_data: {df_tweets_gen.shape[0]}")

    # concatenate DataFrames
    df_tweets = pd.concat([df_tweets_air, df_tweets_gen], ignore_index=True)

    # drop rows/examples where either no text could be fetched or no sentiment was assigned
    print("\nRows with nan/missing entries:")
    print(df_tweets.isnull().sum())
    df_tweets.dropna(axis = 0, how = "any", inplace = True)
    if shuffle == True:
        df_tweets = df_tweets.sample(frac=1)  # shuffle dataframe
    else: pass

    # One-hot encode sentiments
    y = pd.get_dummies(df_tweets["category"])
    return df_tweets, y

def clean_text(text, stopwords = False, stemming = False):
    # takes "raw" text as input and returns a "clean" text
    text = re.sub(r"http\S+", "", text)             # text remove hyperlinks
    text = re.sub(r"@\S+", "", text)                # text remove @mentions
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)       # remove non letters
    text = re.sub(r"^RT[\s]+", "", text)            # remove retweet text "RT"
    text = text.lower()                             # lower case
    text = " ".join(text.split()) # for legibility, avoid having multiple spaces after another
    words = text.split()
    # remove stopwords
    if stopwords == True:
        words = [w for w in words if w not in stopwords.words("english")]
    else: pass
    # apply stemming
    if stemming == True:
        words = [PorterStemmer().stem(w) for w in words]
    else: pass
    text = " ".join(words)
    # return list
    return text

def tokenize_and_pad(texts, vocab_size, max_length):
    # takes "clean" texts as input and returns (and saves) tokenized and padded sequences
    # tokenizing:
    ## assign vocab_size-1 most frequent words numbers from 1 to vocab_size-1
    ## with 1 as highest and vocab_size-1 as lowest frequented word
    tokenizer = Tokenizer(num_words = vocab_size)
    tokenizer.fit_on_texts(texts)
    texts_t = tokenizer.texts_to_sequences(texts)
    # padding:
    ## Add 0 token(s) to sequences shorter than max_length and truncate sequences
    ## longer than max_length
    texts_tp = pad_sequences(texts_t, maxlen = max_length, padding = "post")
    #possible different approach
    #vectorizer = TextVectorization(max_tokens=vocab_size, output_mode = 'int', output_sequence_length = max_length)
    #vectorizer.adapt(texts)
    #return texts.map(vectorizer), vectorizer
    # saving
    # save tokenizer
    with open(r'model_data_keras_embedding\tokenizer_save.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save padder/max_length
    with open(r'model_data_keras_embedding\padder_save.pickle', 'wb') as handle:
        pickle.dump(max_length, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return texts_tp, tokenizer
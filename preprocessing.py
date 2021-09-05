# imports
# GENERAL
import sys   # for displaying a progress-count
import io    # for saving output
import numpy as np
import pandas as pd
# for PREPROCESSING
import re    # RegEx for removing non-letter characters
import nltk  # natural language tool kit
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def clean_text(text):
    # takes "raw" text as input and returns a "clean" text
    text = text.lower()                             # lower case
    text = re.sub(r"http\S+", "", text)             # text remove hyperlinks
    text = re.sub(r"@\S+", "", text)                # text remove @mentions
    ###### EMOJIS
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)       # remove non letters ###################### exclude numbers, in general: how to handle shorts (you've)
    #text = re.sub(r"^RT[\s]+", "", text)            # remove retweet text "RT"
    text = " ".join(text.split()) # for legibility, avoid having multiple spaces after another
    #words = text.split()
    # remove stopwords
    #words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    #words = [PorterStemmer().stem(w) for w in words]
    #text = " ".join(words)
    # return list
    return text

def tokenize_and_pad(texts, vocab_size, max_length):
    # takes "clean" texts as input and returns tokenized and padded sequences
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
    return texts_tp, tokenizer
    #possible different approach
    #vectorizer = TextVectorization(max_tokens=vocab_size, output_mode = 'int', output_sequence_length = max_length)
    #vectorizer.adapt(texts)
    #return texts.map(vectorizer), vectorizer
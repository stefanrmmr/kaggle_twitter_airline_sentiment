# USE CASE: use twitter API to find most recent relevant tweets for an airline and analyse their average sentiment
# Adrian Brünger, Stefan Rummer, TUM, Python Data Analysis for Engineers, summer 2021

from datetime import timedelta
from icecream import ic
import pandas as pd
import pickle
import tweepy
import emoji
import yaml
import re
from statistics import mode
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from plotting_framework import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# for TensorFlow to work without any interruptions,
# set this environment variable to disable all logging output

checkpoint_filepath = r'model_data_keras_embedding\best_model.hdf5'
# The model weights (that are considered the best) are loaded into the model.
model = load_model(checkpoint_filepath)

# IMPORT TOKENIZER SETTING
with open(r'model_data_keras_embedding\tokenizer_save.pickle', 'rb') as handle_import:
    tokenizer_import = pickle.load(handle_import)  # load tokenizer
    # always use the same tokenizer so that word tokens are not changed

max_len = 33  # model specific parameter, result of fundamental data analysis
# maximal amount of words in a tweet


def tweet_cleanup(tweet_import):

    def extract_emojis(text_import):                # extract a list containing the emojis in twe tweet
        emoji_list = []
        [emoji_list.append(c) for c in text_import if c in emoji.UNICODE_EMOJI['en']]
        emoji_list = list(set(emoji_list))          # REMOVE DUPLICATE emojis in the list
        return emoji_list

    def remove_emojis(text_import):
        regex_pattern = re.compile(pattern="["
                                           u"\U0001F600-\U0001F64F"  # emoticons
                                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                           u"\U00002702-\U000027B0"
                                           u"\U00002702-\U000027B0"
                                           u"\U000024C2-\U0001F251"
                                           u"\U0001f926-\U0001f937"
                                           u"\U00010000-\U0010ffff"
                                           u"\u2640-\u2642"
                                           u"\u2600-\u2B55"
                                           u"\u200d"
                                           u"\u23cf"
                                           u"\u23e9"
                                           u"\u231a"
                                           u"\ufe0f"  # dingbats
                                           u"\u3030"
                                           "]+", flags=re.UNICODE)
        return regex_pattern.sub(' ', text_import)

    emojis_in_text = extract_emojis(tweet_import)
    emoji_string = ' '.join(emj for emj in emojis_in_text)

    text = remove_emojis(tweet_import)              # remove all emojis after extraction
    text = text + emoji_string                      # add every emoji ONCE at the end
    text = emoji.demojize(text)                     # translate emojis into words

    text = text.lower()                             # convert text lower case
    text = re.sub(r"http\S+", "", text)             # text remove hyperlinks
    text = re.sub(r"#", "", text)                   # text remove hashtag symbol
    text = re.sub(r"@\S+", "", text)                # text remove @mentions
    text = ''.join((char for char in text
                    if not char.isdigit()))         # remove all numbers
    text = re.sub(r"'", "", text)                   # remove apostrophes
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)       # remove non letters
    text = re.sub(r"^RT[\s]+", "", text)            # remove retweet text "RT"
    text = ' '.join(text.split())                   # remove multiple white space
    text = text.lstrip()                            # remove space from the left
    return text


def predict_class(tweet):

    # apply tweet cleanup function
    text = tweet_cleanup(tweet)
    print(text)

    sentiment_classes = ['Negative', 'Neutral', 'Positive']  # [0, 1, 2]
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer_import.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, maxlen=max_len, padding='post')  # value=0, dtype='int32',
    # Do the prediction using the loaded model
    print(xt)
    # print(f"Sentiments: {model.predict(xt)}")


    yt = model.predict(xt).argmax(axis=1)
    print(yt)
    # Print the predicted sentiment
    # print('SENTIMENT prediction: ', sentiment_classes[mode(yt)])  # TODO why does this yield multiple padded sequences?
    print('The predicted sentiment is', sentiment_classes[yt[0]])  # TODO WTF is this why choose the first entry

    return model.predict(xt)


predict_class("I hate when I have to call and wake people up")

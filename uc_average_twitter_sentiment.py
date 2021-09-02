# USECASE: use twitter API to find most recent relevant tweets for an airline and analyse their average sentiment
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

from datetime import timedelta
from icecream import ic
import pandas as pd
import pickle
import tweepy
import emoji
import yaml
import re
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

from plotting_framework import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# for TensorFlow to work without any interruptions,
# set this environment variable to disable all logging output

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory

# import credentials and access key for twitter access from yaml
with open(r"twitter_acc\twitter_acc_config.yml") as twitter_acc_yml:
    twitter_acc = yaml.load(twitter_acc_yml, Loader=yaml.FullLoader)

consumer_key = twitter_acc.get("Twitter_API_ConsumerKey")
consumer_secret = twitter_acc.get("Twitter_API_ConsumerSecret")
access_token = twitter_acc.get("Twitter_API_AccessToken")
access_secret = twitter_acc.get("Twitter_API_AccessSecret")
consumer_bearer_token = twitter_acc.get("Twitter_API_BearerToken")

# authentication with twitter via OAuthHandler
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
twitter_api = tweepy.API(auth, wait_on_rate_limit=True)

max_len = 52  # model specific parameter, result of fundamental data analysis
# maximal amount of words in a tweet
checkpoint_filepath = r'model_data_keras_embedding\best_model.hdf5'
# The model weights (that are considered the best) are loaded into the model.
model = load_model(checkpoint_filepath)

with open(r'model_data_keras_embedding\tokenizer_save.pickle', 'rb') as handle_import:
    tokenizer_import = pickle.load(handle_import)  # load tokenizer
    # always use the same tokenizer so that word tokens are not changed


def predict_class(text):


    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer_import.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    print(f"Sentiments: {model.predict(xt)}")

    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('SENTIMENT prediction: ', sentiment_classes[yt[0]])
    return model.predict(xt)


def tweets_sentiment(twitter_tag, issue_name, n_tweets,
                     language, timeframe, results, plot):

    lst_text = []
    lst_id = []
    lst_time = []
    lst_retweets = []
    lst_sentpos = []
    lst_sentneg = []
    lst_sentntr = []
    lst_sentdiff = []

    for tweet in tweepy.Cursor(
            twitter_api.search,
            q=twitter_tag,  # + " -filter:retweets"
            # FILTERING for original tweets causes distorted results
            lang=language,  # select only tweets written in english language
            since=(datetime.now() - timedelta(int(abs(timeframe)))).strftime('%Y-%m-%d'),
            until=(datetime.now() + timedelta(1)).strftime('%Y-%m-%d'),
            result_type=results, tweet_mode="extended", include_entities=False
            ).items(int(n_tweets)):

        tweet_time = tweet.created_at
        tweet_id = tweet.id
        tweet_retweets = tweet.retweet_count

        tweet_text = tweet.full_text.replace("\n", " ")         # text remove line breaks
        tweet_text = re.sub(r"http\S+", "", tweet_text)         # text remove hyperlinks
        tweet_text = re.sub(r'#', '', tweet_text)               # text remove hashtag symbol
        tweet_text = re.sub(r"@\S+", "", tweet_text)            # text remove @mentions
        tweet_text = re.sub(r'^RT[\s]+', '', tweet_text)        # remove retweet text "RT"
        tweet_text = tweet_text.lower()                         # make all text lower case
        tweet_text = emoji.demojize(tweet_text)                 # translate emojis

        # use the model trained to evaluate the tweet text, receive sentiment scores
        tweet_sent = predict_class(tweet_text)
        tweet_score_pos = float(tweet_sent[2])
        tweet_score_neg = float(tweet_sent[0])
        tweet_score_ntr = float(tweet_sent[1])

        lst_text.append(tweet_text)
        lst_id.append(tweet_id)
        lst_time.append(tweet_time)
        lst_retweets.append(tweet_retweets)
        lst_sentpos.append(tweet_score_pos)   # positivity sentiment score
        lst_sentneg.append(tweet_score_neg)   # negativity sentiment score
        lst_sentntr.append(tweet_score_ntr)   # neutrality sentiment score
        lst_sentdiff.append(tweet_score_pos - tweet_score_neg)

        print(f"tweepy {tweet_id} |{tweet_time} |"
              f"POS[{'{:0.4f}'.format(tweet_score_pos)}], "
              f"NEG[{'{:0.4f}'.format(tweet_score_neg)}], "
              f"NTR[{'{:0.4f}'.format(tweet_score_ntr)}] |"
              f"{tweet_text}")


    tweets_df = pd.DataFrame(list(zip(lst_id, lst_time, lst_retweets,
                                      lst_text, lst_sentpos, lst_sentneg, lst_sentntr, lst_sentdiff)),
                             columns=['tweet_id', 'tweet_time', 'tweet_retweets', 'tweet_text',
                                      'tweet_sentpos', 'tweet_sentneg', 'tweet_sentntr', 'tweet_sentdiff'])

    ic(tweets_df)  # merge lists into pandas dataframe

    mean_sentpos = round(statistics.mean(tweets_df["tweet_sentpos"]), 4)
    mean_sentneg = round(statistics.mean(tweets_df["tweet_sentneg"]), 4)
    mean_sentntr = round(statistics.mean(tweets_df["tweet_sentntr"]), 4)
    mean_sentiments = [mean_sentpos, mean_sentneg, mean_sentntr]

    n_tweets_analyzed = len(tweets_df)
    first_tweet_time = lst_time[0]
    last_tweet_time = lst_time[len(lst_time)-1]

    if plot:  # optionally generate a plot and save the png
        plot_box_sentiment(lst_sentdiff,
                           mean_sentiments,
                           f"first tweet {first_tweet_time}\n"
                           f"last tweet {last_tweet_time}\n"
                           f"search tag \"{twitter_tag}\"\n"
                           f"{n_tweets_analyzed} analyzed tweets [finBERT]")

    # output the resulting mean sentiment for the analysis
    print(f"\nTwitter Sentiment Analysis - MEAN Result: "
          f"POS[{'{:0.4f}'.format(mean_sentiments[0])}],"
          f" NEG[{'{:0.4f}'.format(mean_sentiments[1])}],"
          f" NTR[{'{:0.4f}'.format(mean_sentiments[2])}]")
    print(f"For \"{issue_name}\" a total of [{n_tweets_analyzed}] tweets have been found and analyzed.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
    return mean_sentiments, n_tweets_analyzed, first_tweet_time, last_tweet_time


# TODO test the script output for given input values
# tweets_sentiment("Siemens Energy", "Siemens Energy", 100, "en", 1, "recent", True)
#tweets_sentiment("$HIMS", "HIMS", 100, "en", 1, "recent", True)
#tweets_sentiment("$BZN OR Baozun", "Baozun", 100, "en", 1, "recent", True)
# tweets_sentiment("$TDOC OR Teladoc", "Teladoc", 250, "en", 1, "recent", True)
# tweets_sentiment("@Siemens_Energy", "Siemens Energy", 100, "en", 1, "recent", True)
# tweets_sentiment("rip", "rip", 50, "en", 1, "recent", True)
# tweets_sentiment("$NLLSF OR NEL", "NEL", 250, "en", 2, "recent", True)
tweets_sentiment("delta", "Delta Airlines", 100, "en", 1, "popular", True)
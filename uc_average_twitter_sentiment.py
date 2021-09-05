# USE CASE: use twitter API to find most recent relevant tweets for an airline and analyse their average sentiment
# Adrian Brünger, Stefan Rummer, TUM, Python Data Analysis for Engineers, summer 2021

from keras.models import load_model
from datetime import timedelta
from icecream import ic
import pandas as pd
import pickle
import tweepy
import emoji
import yaml
import re

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

max_len = 33  # model specific parameter, result of fundamental data analysis
# maximal amount of words in a tweet

checkpoint_filepath = r'model_data_final\best_model.hdf5'
# The model weights (that are considered the best) are loaded into the model.
model = load_model(checkpoint_filepath)

with open(r'model_data_final\tokenizer_save.pickle', 'rb') as handle_import:
    tokenizer_import = pickle.load(handle_import)  # load tokenizer
    # always use the same tokenizer so that word tokens are not changed


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


def predict_sentiment(text_input):

    text_list = [tweet_cleanup(text_input)]  # Transforms text to a sequence of integers
    sequence = tokenizer_import.texts_to_sequences(text_list)  # [[3, 157, 24, 201, 7, 156]] SHAPE
    sequence = sequence[0]                                     # [3, 157, 24, 201, 7, 156]   SHAPE

    sequence = sequence + [0] * (max_len - len(sequence))
    # pad_sequences(output1, maxlen=max_len, padding='post')
    df_input = pd.DataFrame(sequence)  # create dataframe
    df_input_t = df_input.transpose()  # transpose dataframe

    sentiment_scores = model.predict(df_input_t)
    sentiment_scores = list(sentiment_scores[0])
    # print(f"\nSENTIMENT Scores: {sentiment_scores}")

    # sentiment_classes = ['Negative', 'Neutral', 'Positive']
    # sentiment_detected = sentiment_scores.index(max(sentiment_scores))
    # print(f'SENTIMENT Prediction: {sentiment_classes[sentiment_detected]}')

    return sentiment_scores  # [neg, neut, pos]


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

        # use the model trained to evaluate the tweet text, receive sentiment scores
        tweet_sent = predict_sentiment(tweet_text)
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
              f"NEG[{'{:0.4f}'.format(tweet_score_neg)}], "
              f"NTR[{'{:0.4f}'.format(tweet_score_ntr)}], "
              f"POS[{'{:0.4f}'.format(tweet_score_pos)}] | "
              f"{tweet_text}")

    tweets_df = pd.DataFrame(list(zip(lst_id, lst_time, lst_retweets,
                                      lst_text, lst_sentpos, lst_sentneg, lst_sentntr, lst_sentdiff)),
                             columns=['tweet_id', 'tweet_time', 'tweet_retweets', 'tweet_text',
                                      'tweet_sentpos', 'tweet_sentneg', 'tweet_sentntr', 'tweet_sentdiff'])

    ic(tweets_df)  # merge lists into pandas dataframe

    mean_sentpos = round(statistics.mean(tweets_df["tweet_sentpos"]), 4)
    mean_sentneg = round(statistics.mean(tweets_df["tweet_sentneg"]), 4)
    mean_sentntr = round(statistics.mean(tweets_df["tweet_sentntr"]), 4)
    mean_sentiments = [mean_sentneg, mean_sentntr, mean_sentpos]

    n_tweets_analyzed = len(tweets_df)
    last_tweet_time = lst_time[0]
    first_tweet_time = lst_time[len(lst_time)-1]

    if plot:  # optionally generate a plot and save the png
        plot_box_sentiment(lst_sentdiff,
                           mean_sentiments,
                           issue_name,
                           f"first tweet {first_tweet_time}\n"
                           f"last tweet {last_tweet_time}\n"
                           f"search tag \"{twitter_tag}\"\n"
                           f"{n_tweets_analyzed} analyzed tweets [keras-LSTM]")

    # output the resulting mean sentiment for the analysis
    print(f"\nTwitter Sentiment Analysis - MEAN Result: "
          f" NEG[{'{:0.4f}'.format(mean_sentiments[0])}],"
          f" NTR[{'{:0.4f}'.format(mean_sentiments[1])}],"
          f" POS[{'{:0.4f}'.format(mean_sentiments[2])}]")
    print(f"For \"{issue_name}\" a total of [{n_tweets_analyzed}] tweets have been found and analyzed.")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n")
    return mean_sentiments, n_tweets_analyzed, first_tweet_time, last_tweet_time


# TODO test the script output for given input values
tweets_sentiment("@delta", "Delta Airlines", 250, "en", 1, "recent", True)

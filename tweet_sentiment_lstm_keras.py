# FINAL MODEL for Twitter tweet text sentiment analysis
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

import pickle
import emoji
import pandas as pd
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split

import keras.backend as k
from keras.models import load_model
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, SpatialDropout1D
from keras.layers import Bidirectional, LSTM, Dense  # Dropout

import tensorflow as tf
from tensorflow.keras.optimizers import Adam  # SGD, RMSprop

from plotting_framework import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# for TensorFlow to work without any interruptions,
# set this environment variable to disable all logging output

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory

# DEFINE MODEL CHARACTERISTICS
vocabulary_size = 3000   # TODO HYPER PARAMETER
embedding_size = 32      # TODO HYPER PARAMETER
epochs = 20              # TODO HYPER PARAMETER
learning_rate = 0.0005   # TODO HYPER PARAMETER
#momentum = 0.0           # TODO HYPER PARAMETER
batch_size = 64          # TODO HYPER PARAMETER


def tweet_cleanup(tweet_import):

    def extract_emojis(text_import):                # extract a list containing the emojis in twe tweet
        emoji_list = []
        [emoji_list.append(c) for c in text_import if c in emoji.UNICODE_EMOJI.keys()] # ["en"]
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


def tokenize_pad_sequences(tweets):

    # This function tokenize the input text into sequences of integers and pads each sequence
    # Text tokenization with max amount of vocab_size word token ids
    tokenizer_padseq = Tokenizer(num_words=vocabulary_size, split=' ')
    tokenizer_padseq.fit_on_texts(tweets)
    # print(tokenizer_padseq.word_index)

    # Transforms text to a sequence of integers
    output = tokenizer_padseq.texts_to_sequences(tweets)
    # Pad sequences to the same length
    output = pad_sequences(output, maxlen=max_len, padding='post')  # truncating='post',

    return output, tokenizer_padseq


def f1_score(precision_val, recall_val):
    f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + k.epsilon())
    return f1_val


def predict_sentiment(text_input):

    with open(r'model_data_final\tokenizer_save.pickle', 'rb') as handle_import:
        tokenizer_import = pickle.load(handle_import)  # load tokenizer

    text_list = [tweet_cleanup(text_input)]  # Transforms text to a sequence of integers
    sequence = tokenizer_import.texts_to_sequences(text_list)  # [[3, 157, 24, 201, 7, 156]] SHAPE
    sequence = sequence[0]                                     # [3, 157, 24, 201, 7, 156]   SHAPE

    sequence = sequence + [0] * (max_len - len(sequence))
    # pad_sequences(output1, maxlen=max_len, padding='post')
    df_input = pd.DataFrame(sequence)  # create dataframe
    df_input_t = df_input.transpose()  # transpose dataframe

    sentiment_scores = model.predict(df_input_t)
    sentiment_scores = list(sentiment_scores[0])
    print(f"\nSENTIMENT Scores: {sentiment_scores}")

    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    sentiment_detected = sentiment_scores.index(max(sentiment_scores))
    print(f'SENTIMENT Prediction: {sentiment_classes[sentiment_detected]}')

    return sentiment_scores


print("\n_______DAML_Twitter_Sentiment________\n")


# IMPORT DATA TWEETS: Airlines
df_tweets_air_full = pd.read_csv('tweets_data/Tweets_airlines.csv')
print(df_tweets_air_full.info(), "\n")

df_tweets_air = df_tweets_air_full.copy()
df_tweets_air = df_tweets_air.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
df_tweets_air['category'] = df_tweets_air['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
df_tweets_air = df_tweets_air[['category', 'clean_text']]
"""# IMPORT DATA TWEETS: General
df_tweets_gen = pd.read_csv('tweets_data/Tweets_general.csv')
df_tweets_gen = df_tweets_gen[['category', 'clean_text']]
# COMBINE DATASETS for large amount of data, increase accuracy"""

df_tweets = df_tweets_air
# df_tweets = pd.concat([df_tweets_air, df_tweets_gen], ignore_index=True)

df_tweets.isnull().sum()                # Check for missing data
df_tweets.dropna(axis=0, inplace=True)  # Drop missing rows
print(df_tweets.head(10), "\n")         # output first ten tweet df entries BEFORE PREPROCESSING

# Apply data processing to each tweet in df_tweets['clean_text']
seq_count = 1
tweet_data_size = len(df_tweets['clean_text'])
print("\n===================================================")
for index in range(tweet_data_size):
    df_tweets.at[index, 'clean_text'] = tweet_cleanup(df_tweets.iloc[index]['clean_text'])
    sys.stdout.write(f"\rPreprocessing tweet texts: {str(seq_count).zfill(6)}/{tweet_data_size}")
    sys.stdout.flush()
    seq_count += 1

print("\n", df_tweets.head(10), "\n")         # output first ten tweet df entries AFTER PREPROCESSING

# DYNAMICALLY change the max_len parameter
max_len = max([len(tweet.split()) for tweet in df_tweets['clean_text']])
print(f"\nMax number of words expected"
      f" in a processed tweet: {max_len} \n")


print('Before Tokenization & Padding \n', df_tweets['clean_text'][2])
# TOKENIZE and PAD the list of word arrays

X_tweets_list, tokenizer = tokenize_pad_sequences(df_tweets['clean_text'].values)
print('After Tokenization & Padding \n', X_tweets_list[2])


with open(r'model_data_final\tokenizer_save.pickle', 'wb') as handle:  # save tokenizer
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# TARGET vector ONE HOT ENCODING (3dummy variables)
y_categories = pd.get_dummies(df_tweets['category'])

# TRAIN VALIDATION SPLIT (60% train, 20% valid, 20% test)
X_train, X_test, y_train, y_test = \
    train_test_split(X_tweets_list, y_categories, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.25, random_state=1)

print(f"SHAPE X_train | rows: {X_train.shape[0]} cols: {X_train.shape[1]}")
print(f"SHAPE X_valid | rows: {X_val.shape[0]} cols: {X_val.shape[1]}")
print(f"SHAPE X_test  | rows: {X_test.shape[0]} cols: {X_test.shape[1]}")
print(f"SHAPE Y_train | rows: {y_train.shape[0]} cols: {y_train.shape[1]}")
print(f"SHAPE Y_valid | rows: {y_val.shape[0]} cols: {y_val.shape[1]}")
print(f"SHAPE Y_test  | rows: {y_test.shape[0]} cols: {y_test.shape[1]}\n")

adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
# OPTIMIZERS, use standard settings for adam optimizer

model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len, trainable=True))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(32)))  # , dropout=0.2, recurrent_dropout=0.2
# model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

# PLOT model structure and layers
# tf.keras.utils.plot_model(model, show_shapes=True)
print(model.summary())  # OUTPUT model information

model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy', Precision(), Recall()])

# AUTOMATIC RESTORATION of optimal model configuration AFTER training completed
# RESTORE the OPTIMAL NN WEIGHTS from when val_loss was minimal (epoch nr.)
# SAVE model weights at the end of every epoch, if these are the best so far
checkpoint_filepath = r'model_data_final\best_model.hdf5'
model_checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_filepath, patience=3, verbose=1,
                    save_best_only=True, monitor='val_accuracy', mode='max')

# apply model to training data and store history information
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    callbacks=[model_checkpoint_callback])

# PLOT ACCURACY and LOSS evolution
plot_training_hist(history, epochs)

# The model weights (that are considered the best) are loaded into the model.
model = load_model(checkpoint_filepath)

# Evaluate model on the VALIDATION SET  METRICS
loss, accuracy, precision, recall = model.evaluate(X_val, y_val, verbose=0)
print("\n___________________________________________________")
print('VALIDATION Dataset Loss      : {:.4f}'.format(loss))
print('VALIDATION Dataset Accuracy  : {:.4f}'.format(accuracy))
print('VALIDATION Dataset Precision : {:.4f}'.format(precision))
print('VALIDATION Dataset Recall    : {:.4f}'.format(recall))
print('VALIDATION Dataset F1 Score  : {:.4f}'.format(f1_score(precision, recall)))
print("===================================================")

# Evaluate model on the TEST SET  METRICS
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print("\n___________________________________________________")
print('TEST Dataset Loss      : {:.4f}'.format(loss))
print('TEST Dataset Accuracy  : {:.4f}'.format(accuracy))
print('TEST Dataset Precision : {:.4f}'.format(precision))
print('TEST Dataset Recall    : {:.4f}'.format(recall))
print('TEST Dataset F1 Score  : {:.4f}'.format(f1_score(precision, recall)))
print("===================================================")

# PLOT CONFUSION MATRIX for TEST Dataset
plot_confusion_matrix(model, X_test, y_test)

print(predict_sentiment("I love this, best flight ever"))
print(predict_sentiment("I had a perfect experience 324324 "))
print(predict_sentiment("You should fire this bad service"))
print(predict_sentiment("My flight was hours delayed"))
print(predict_sentiment("The flight was perfect!"))

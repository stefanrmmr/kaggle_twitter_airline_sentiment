# Kaggle-Inspired PIPELINE for Sentiment Analysis
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

import nltk
import pickle
import pandas as pd

from nltk.stem.porter import *
from nltk.corpus import stopwords

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

import keras.backend as k
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.metrics import Precision, Recall
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, LSTM, Dense, Dropout

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from plotting_framework import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# for TensorFlow to work without any interruptions,
# set this environment variable to disable all logging output

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory


# DEFINE MODEL CHARACTERISTICS
vocabulary_size = 5000  # TODO HYPER PARAMETER
embedding_size = 64     # TODO HYPER PARAMETER
epochs = 30             # TODO HYPER PARAMETER
learning_rate = 0.1     # TODO HYPER PARAMETER
momentum = 0.8          # TODO HYPER PARAMETER
batch_size = 32         # TODO HYPER PARAMETER


def tweet_to_words(tweet):

    text = tweet.lower()                            # lower case
    text = re.sub(r"http\S+", "", text)             # text remove hyperlinks
    text = re.sub(r"#", "", text)                   # text remove hashtag symbol
    text = re.sub(r"@\S+", "", text)                # text remove @mentions
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)       # remove non letters
    text = re.sub(r"^RT[\s]+", "", text)            # remove retweet text "RT"

    words = text.split()        # tokenize

    # TODO STOPWORDS change this to exclude words from a whitelist (no, not, doesnt, etc...)
    # words = [w for w in words if w not in stopwords.words("english")]
    # TODO STEMMER maybe apply stemming to split into word stems (generalization)
    # words = [PorterStemmer().stem(w) for w in words]
    return words


def tokenize_pad_sequences(text):

    # This function tokenize the input text into sequences of integers
    # and then pads each sequence to the same length

    # Text tokenization
    tokenizer_padseq = Tokenizer(num_words=vocabulary_size, lower=True, split=' ')
    tokenizer_padseq.fit_on_texts(text)
    # Transforms text to a sequence of integers
    output = tokenizer_padseq.texts_to_sequences(text)
    # Pad sequences to the same length
    output = pad_sequences(output, padding='post', maxlen=max_len)
    # return sequences
    return output, tokenizer_padseq


def f1_score(precision_val, recall_val):
    f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + k.epsilon())
    return f1_val


def predict_class(text):
    # Function to predict sentiment class of the passed text

    sentiment_classes = ['Negative', 'Neutral', 'Positive']

    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('SENTIMENT prediction: ', sentiment_classes[yt[0]])


print("\n_______DAML_Twitter_Sentiment________\n")

df_tweets = pd.read_csv('tweets_data/Tweets.csv')
print(df_tweets.info())
df_tweets = df_tweets.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
df_tweets['category'] = df_tweets['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
df_tweets = df_tweets[['category', 'clean_text', 'airline']]

df_tweets.isnull().sum()                # Check for missing data
df_tweets.dropna(axis=0, inplace=True)  # Drop missing rows
print(df_tweets.head(10))  # output first ten tweet df entries

# Apply data processing to each tweet
X_tweets = list(map(tweet_to_words, df_tweets['clean_text']))

# DYNAMICALLY change the max_len parameter
max_len = 0
for entry in X_tweets:
    if len(entry) > max_len:
        max_len = len(entry)
max_len = int(max_len * 1.1)
# add 10% extra to accommodate deviations
print(f"\nMax number of words expected"
      f" in a processed tweet: {max_len} \n")






# TODO TURN categories into one hot encoding?
# TODO tokenisation and padding (each word becomes a special id)
# TODO process only the top 5000 words (vocab_size)
# TODO pad the sequences to have the same length

# this results in having a database where each entry in a sequence can be one of 5000 values (words)
# TODO reduce this dimension to a lower dimension
# the length of the padded list does not change, but the dimensionality of each entry does
# this dimension reduction also bares information on which words are related

# TODO processes used to achieve this form of vectorization/dim reduction
# glove?
# word2vec
# embedding layer in keras




# Encode target labels
label_encoder = LabelEncoder()
Y = label_encoder.fit_transform(df_tweets['category'])




"""y = pd.get_dummies(df_tweets['category'])
# TRAIN TEST SPLIT (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# TRAIN VALIDATION SPLIT (60% train, 20% valid, 20% test)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)"""

# TODO wtf does this do?
# Tweets have already been preprocessed hence dummy function will be passed in
count_vector = CountVectorizer(max_features=vocabulary_size,
                               preprocessor=lambda x: x,
                               tokenizer=lambda x: x)


"""X_train = count_vector.fit_transform(X_train).toarray()  # NORMALIZATION Fit the training data
X_test = count_vector.transform(X_test).toarray()        # NORMALIZATION Transform testing data"""


X_tweets, tokenizer = tokenize_pad_sequences(df_tweets['clean_text'])

with open(r'model_data\tokenizer_save.pickle', 'wb') as handle:  # save tokenizer
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(r'model_data\tokenizer_save.pickle', 'rb') as handle:  # load tokenizer
    tokenizer = pickle.load(handle)

y_categories = pd.get_dummies(df_tweets['category'])







# TRAIN VALIDATION SPLIT (60% train, 20% valid, 20% test)
X_train, X_test, y_train, y_test = \
    train_test_split(X_tweets, y_categories, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print(f"SHAPE X_train: {X_train.shape}")
print(f"SHAPE X_valid: {X_val.shape}")
print(f"SHAPE X_test : {X_test.shape}\n")
print(f"SHAPE Y_train: {y_train.shape}")
print(f"SHAPE Y_valid: {y_val.shape}")
print(f"SHAPE Y_test : {y_test.shape}\n")

# OPTIMIZERS
sgd = SGD(lr=learning_rate, momentum=momentum,
          decay=(learning_rate/epochs), nesterov=False)

# BUILD the model
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

# tf.keras.utils.plot_model(model, show_shapes=True)
print(model.summary())  # OUTPUT model information

model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy', Precision(), Recall()])

# AUTOMATIC RESTORATION of optimal model configuration AFTER training completed
# RESTORE the OPTIMAL NN WEIGHTS from when val_loss was minimal (epoch nr.)
# SAVE model weights at the end of every epoch, if these are the best so far
checkpoint_filepath = r'model_data\best_model.hdf5'
model_checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_filepath, patience = 3, verbose=1,
                    save_best_only=True, monitor='val_loss', mode='min')

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


# TODO test certain phrases to evaluate on the model performance
# predict_class("I've really enjoyed this flight to Cuba.")
# predict_class("the flight was perfect ")
# predict_class("This journey was a pleasure!")

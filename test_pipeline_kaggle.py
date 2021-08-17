# Test Pipeline inspired by kaggle code

import nltk
import pickle
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

import keras.backend as k
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall

import tensorflow as tf
from tensorflow.keras.optimizers import SGD

from plotting_framework import *

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # TODO wtf but necessary for tensorflow to work


# DEFINE ESSENTIAL VARIABLES
max_len = 50
vocabulary_size = 5000  # HYPER PARAMETER  # TODO SWEEP this!!

# DEFINE MODEL CHARACTERISTICS
embedding_size = 64     # HYPER PARAMETER  # TODO SWEEP this!!
epochs = 30             # HYPER PARAMETER  # TODO SWEEP this!!
learning_rate = 0.1     # HYPER PARAMETER  # TODO SWEEP this!!
momentum = 0.8          # HYPER PARAMETER  # TODO SWEEP this!!
batch_size = 32         # HYPER PARAMETER  # TODO SWEEP this!!
decay_rate = learning_rate / epochs

print("\n\n_______DSML_Twitter_Sentiment________\n\n")

df_tweets = pd.read_csv('tweets_data/Tweets.csv')
print(df_tweets.info())
df_tweets = df_tweets.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
df_tweets['category'] = df_tweets['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
df_tweets = df_tweets[['category', 'clean_text', 'airline']]

df_tweets.isnull().sum()                # Check for missing data
df_tweets.dropna(axis=0, inplace=True)  # Drop missing rows
print(df_tweets.head(10))  # output first ten tweet df entries


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

    # This function tokenize the input text into sequences of integers and then
    # pad each sequence to the same length

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
    max_len_val = 50

    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len_val)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('SENTIMENT prediction: ', sentiment_classes[yt[0]])


# print("\nOriginal tweet ->", df_tweets['clean_text'][0])
# print("\nProcessed tweet ->", tweet_to_words(df_tweets['clean_text'][0]), "\n")

# Apply data processing to each tweet
X = list(map(tweet_to_words, df_tweets['clean_text']))

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


X, tokenizer = tokenize_pad_sequences(df_tweets['clean_text'])

with open('tokenizer.pickle', 'wb') as handle:  # save tokenizer
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('tokenizer.pickle', 'rb') as handle:  # load tokenizer
    tokenizer = pickle.load(handle)

y = pd.get_dummies(df_tweets['category'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
# TRAIN VALIDATION SPLIT (60% train, 20% valid, 20% test)
print('Train Set ->', X_train.shape, y_train.shape)
print('Validation Set ->', X_val.shape, y_val.shape)
print('Test Set ->', X_test.shape, y_test.shape)

# BUILD the model
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

tf.keras.utils.plot_model(model, show_shapes=True)
print(model.summary())  # OUTPUT model information

model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy', Precision(), Recall()])

# apply model to training data and store history information
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=batch_size, epochs=epochs, verbose=1)

plot_training_hist(history, epochs)
plot_confusion_matrix(model, X_test, y_test)

# Evaluate model on the test set METRICS
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print("\n\n___________________________________________________")
print('Accuracy  : {:.4f}'.format(accuracy))
print('Precision : {:.4f}'.format(precision))
print('Recall    : {:.4f}'.format(recall))
print('F1 Score  : {:.4f}'.format(f1_score(precision, recall)))
print("___________________________________________________")

# Save the model architecture & the weights
model.save('best_model.h5')             # SAVE the best model
model = load_model('best_model.h5')     # RELOAD the saved model

# TODO test certain phrases to evaluate on the model performance
predict_class("I've really enjoyed this flight to Cuba, however the food was unsatisfying.")
predict_class("I love my Mama")
predict_class("This journey was a pleasure!")

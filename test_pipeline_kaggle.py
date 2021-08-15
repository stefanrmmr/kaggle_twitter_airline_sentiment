
import pandas as pd  # data processing

import nltk  # natural language tool kit
import pickle

from nltk.corpus import stopwords
from nltk.stem.porter import *

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

# import tensorflow as tf
import keras.backend as k

from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall

from tensorflow.keras.optimizers import SGD

nltk.download("stopwords")
pd.options.plotting.backend = "plotly"

# Load Tweet dataset
df_tweets = pd.read_csv('/tweets_data/Tweets.csv')
df_tweets = df_tweets.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
df_tweets['category'] = df_tweets['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
df_tweets = df_tweets[['category', 'clean_text']]

# Output first five rows
df_tweets.head()
# Check for missing data
df_tweets.isnull().sum()
# drop missing rows
df_tweets.dropna(axis=0, inplace=True)

# Map tweet categories
df_tweets['category'] = df_tweets['category'].map({-1.0: 'Negative', 0.0: 'Neutral', 1.0: 'Positive'})

# Output first five rows
df_tweets.head()


def tweet_to_words(tweet):

    # convert to lowercase
    text = tweet.lower()
    # remove non letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # tokenize
    words = text.split()
    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    words = [PorterStemmer().stem(w) for w in words]
    # return list
    return words


print("\nOriginal tweet ->", df_tweets['clean_text'][0])
print("\nProcessed tweet ->", tweet_to_words(df_tweets['clean_text'][0]))

# Apply data processing to each tweet
X = list(map(tweet_to_words, df_tweets['clean_text']))

# Encode target labels
le = LabelEncoder()
Y = le.fit_transform(df_tweets['category'])

y = pd.get_dummies(df_tweets['category'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


vocabulary_size = 5000

# Tweets have already been preprocessed hence dummy function will be passed in
# to preprocessor & tokenizer step
count_vector = CountVectorizer(max_features=vocabulary_size,
                               preprocessor=lambda x: x,
                               tokenizer=lambda x: x)
# tfidf_vector = TfidfVectorizer(lowercase=True, stop_words='english')

# Fit the training data
X_train = count_vector.fit_transform(X_train).toarray()
# Transform testing data
X_test = count_vector.transform(X_test).toarray()

max_words = 5000
max_len = 50


def tokenize_pad_sequences(text):

    # This function tokenize the input text into sequences of integers and then
    # pad each sequence to the same length

    # Text tokenization
    tokenizer_padseq = Tokenizer(num_words=max_words, lower=True, split=' ')
    tokenizer_padseq.fit_on_texts(text)
    # Transforms text to a sequence of integers
    output = tokenizer_padseq.texts_to_sequences(text)
    # Pad sequences to the same length
    output = pad_sequences(output, padding='post', maxlen=max_len)
    # return sequences
    return output, tokenizer_padseq


print('Before Tokenization & Padding \n', df_tweets['clean_text'][0])
X, tokenizer = tokenize_pad_sequences(df_tweets['clean_text'])
print('After Tokenization & Padding \n', X[0])

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

y = pd.get_dummies(df_tweets['category'])


def f1_score(precision_val, recall_val):
    f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + k.epsilon())
    return f1_val


# define essential variables
vocab_size = 5000   # HYPERPARAMETER
embedding_size = 32
epochs = 20
learning_rate = 0.1
decay_rate = learning_rate / epochs
momentum = 0.8

# BUILD the model
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length=max_len))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

# tf.keras.utils.plot_model(model, show_shapes=True)
print(model.summary())  # OUTPUT model information

model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy', Precision(), Recall()])

# Train model
batch_size = 64
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=batch_size, epochs=epochs, verbose=1)

# Evaluate model on the test set METRICS
loss, accuracy, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print('')
print('Accuracy  : {:.4f}'.format(accuracy))
print('Precision : {:.4f}'.format(precision))
print('Recall    : {:.4f}'.format(recall))
print('F1 Score  : {:.4f}'.format(f1_score(precision, recall)))


def plot_training_hist(history_import):
    # Function to plot history for accuracy and loss

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # first plot
    ax[0].plot(history_import.history['accuracy'])
    ax[0].plot(history_import.history['val_accuracy'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].legend(['train', 'validation'], loc='best')
    # second plot
    ax[1].plot(history_import.history['loss'])
    ax[1].plot(history_import.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend(['train', 'validation'], loc='best')


plot_training_hist(history)


def plot_training_hist(history_import):
    # Function to plot history for accuracy and loss

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    # first plot
    ax[0].plot(history_import.history['accuracy'])
    ax[0].plot(history_import.history['val_accuracy'])
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('epoch')
    ax[0].set_ylabel('accuracy')
    ax[0].legend(['train', 'validation'], loc='best')
    # second plot
    ax[1].plot(history_import.history['loss'])
    ax[1].plot(history_import.history['val_loss'])
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('epoch')
    ax[1].set_ylabel('loss')
    ax[1].legend(['train', 'validation'], loc='best')


plot_training_hist(history)

# Save the model architecture & the weights
model.save('best_model.h5')
print('Best model saved')

# Load model
model = load_model('best_model.h5')


def predict_class(text):
    # Function to predict sentiment class of the passed text

    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50

    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('The predicted sentiment is', sentiment_classes[yt[0]])


predict_class("I've really enjoyed this flight to Cuba however the food was unsatisfying.")

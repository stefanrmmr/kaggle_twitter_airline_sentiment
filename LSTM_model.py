# FINAL MODEL for Twitter tweet text sentiment analysis
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

# imports
# GENERAL
import os
import sys
import pickle
import numpy as np
import pandas as pd
# for PREPROCESSING
from preprocessing import *
from sklearn.model_selection import train_test_split
# for LSTM MODEL
import tensorflow as tf
from tensorflow.keras.optimizers import Adam  # SGD, RMSprop

import keras.backend as k
from keras.models import load_model
from keras.models import Sequential
from keras.metrics import Precision, Recall
from keras.layers import Embedding  # Conv1D, MaxPooling1D
from keras.layers import Bidirectional, LSTM, Dense  # Dropout

#for PLOTTING
from plotting_framework import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# for TensorFlow to work without any interruptions,
# set this environment variable to disable all logging output

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory

def build_LSTM(vocab_size, max_length, embedding_size, embedding_matrix, trainable = True):
    # build lstm model from input params and output compiled model
    #1) build model
    model = Sequential()
    model._name = "LSTM"
    if trainable == True: # add trainable embedding layer
        model.add(Embedding(vocab_size, embedding_size, input_length = max_length, trainable = True))
    else: # use pretrained weights specified by "embedding_matrix"
        model.add(Embedding(vocab_size, embedding_size, weights = [embedding_matrix], input_length = max_length, trainable = False))
    # model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    # model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)))
    # model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    # tf.keras.utils.plot_model(model, show_shapes=True)
    print(model.summary())  # OUTPUT model information
    #2) compile model

    learning_rate = 0.0001   # TODO HYPER PARAMETER
    momentum = 0.0           # TODO HYPER PARAMETER

    # optimizers
    """sgd = SGD(lr=learning_rate, momentum=momentum,
            decay=(learning_rate/epochs), nesterov=False)
    rmsprop = RMSprop(learning_rate=learning_rate, rho=0.9, momentum=momentum,
                                epsilon=1e-07, centered=True)"""
    adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

    model.compile(loss='categorical_crossentropy', optimizer=adam,
                metrics=['accuracy', Precision(), Recall()])

    return model

def f1_score(precision_val, recall_val):
    # for evaluating model performance
    f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + k.epsilon())
    return f1_val

def predict_class(model, tokenizer, max_length, text):

    sentiment_classes = np.array(['Negative', 'Neutral', 'Positive'])

    # Transforms text to a sequence of integers using a tokenizer object
    X_t = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    X_tp = pad_sequences(X_t, maxlen = max_length, padding = 'post')
    # Do the prediction using the loaded model
    print(f"Sentiments:\n{model.predict(X_tp)}")

    y_hat = np.array(model.predict(X_tp).argmax(axis=1))
    # Print the predicted sentiment
    print('SENTIMENT prediction:\n', sentiment_classes[y_hat])
    return None ##### TODO

if __name__ == "__main__":

    print("\n_______DAML_Twitter_Sentiment________\n")

    # load tweets
    df_tweets, y = prepare_dataframe(airline = 1, filler_data = 1, shuffle = False)
    # Examples in airline data: 14640
    # Examples in filler_data: 162980
    # airline, filler_data from [0, 1] determine percentage of airline data
    # and additional data used. Set shuffle to False to reproduce outcomes
    print(df_tweets.head())

    # PREPROCESSING
    ## I) Clean tweets/sentences
    print("Cleaning tweets...")
    df_tweets["clean_text"] = df_tweets["text"].map(clean_text)
    print(df_tweets[["clean_text", "category"]].head())

    ## II) Tokenize and pad sequences to yield arrays of integers with uniform length, each integer representing a unique word
    ### define vocabulary size (n-1 most common words that are kept) and maximum length of an encoded sequence
    print("\nPreprocessing parameters:")
    # TODO custom
    vocab_size = 5000 #len(np.unique(np.concatenate(df_tweets["clean_text"].apply(str.split)))) 
    print(f"vocabulary size: {vocab_size}")

    # TODO custom
    max_length = max(df_tweets["clean_text"].apply(str.split).apply(len)) # length of all sequences = length longest sequence
    print(f"sequence length: {max_length}\n")

    X_tp, tokenizer = tokenize_and_pad(df_tweets["clean_text"], vocab_size, max_length)

    # save vocabulary
    vocab = tokenizer.sequences_to_texts([range(0, vocab_size+1)])[0].split()
    #print(vocab)

    # TRAIN VALIDATION SPLIT (60% train, 20% valid, 20% test)
    X_train, X_test, y_train, y_test = \
        train_test_split(X_tp, y, test_size=0.2, random_state=1)
    X_train, X_val, y_train, y_val = \
        train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    print(f"SHAPE X_train: {X_train.shape}")
    print(f"SHAPE X_valid: {X_val.shape}")
    print(f"SHAPE X_test : {X_test.shape}\n")
    print(f"SHAPE Y_train: {y_train.shape}")
    print(f"SHAPE Y_valid: {y_val.shape}")
    print(f"SHAPE Y_test : {y_test.shape}\n")

    # LSTM MODEL

    # define model characteristics
    embedding_size = 32      # TODO HYPER PARAMETER
    epochs = 20              # TODO HYPER PARAMETER
    batch_size = 164         # TODO HYPER PARAMETER
    
    # build model
    lstm_model = build_LSTM(vocab_size, max_length, embedding_size, embedding_matrix = 0, trainable = True)

    # AUTOMATIC RESTORATION of optimal model configuration AFTER training completed
    # RESTORE the OPTIMAL NN WEIGHTS from when val_loss was minimal (epoch nr.)
    # SAVE model weights at the end of every epoch, if these are the best so far
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    checkpoint_filepath = rf'model_data_keras_embedding\best_model_testrun_{time_of_analysis}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.\
        ModelCheckpoint(filepath=checkpoint_filepath, patience=3, verbose=1,
                        save_best_only=True, monitor='val_loss', mode='min')

    # apply model to training data and store history information
    history = lstm_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        batch_size=batch_size, epochs=epochs, verbose=1,
                        callbacks=[model_checkpoint_callback])

    # PLOT ACCURACY and LOSS evolution
    plot_training_hist(history, epochs)

    # The model weights (that are considered the best) are loaded into the model.
    lstm_model = load_model(checkpoint_filepath)

    # Evaluate model on the VALIDATION SET  METRICS
    loss, accuracy, precision, recall = lstm_model.evaluate(X_val, y_val, verbose=0)
    print("\n___________________________________________________")
    print('VALIDATION Dataset Loss      : {:.4f}'.format(loss))
    print('VALIDATION Dataset Accuracy  : {:.4f}'.format(accuracy))
    print('VALIDATION Dataset Precision : {:.4f}'.format(precision))
    print('VALIDATION Dataset Recall    : {:.4f}'.format(recall))
    print('VALIDATION Dataset F1 Score  : {:.4f}'.format(f1_score(precision, recall)))
    print("===================================================")

    # Evaluate model on the TEST SET  METRICS
    loss, accuracy, precision, recall = lstm_model.evaluate(X_test, y_test, verbose=0)
    print("\n___________________________________________________")
    print('TEST Dataset Loss      : {:.4f}'.format(loss))
    print('TEST Dataset Accuracy  : {:.4f}'.format(accuracy))
    print('TEST Dataset Precision : {:.4f}'.format(precision))
    print('TEST Dataset Recall    : {:.4f}'.format(recall))
    print('TEST Dataset F1 Score  : {:.4f}'.format(f1_score(precision, recall)))
    print("===================================================")

    # PLOT CONFUSION MATRIX for TEST Dataset
    plot_confusion_matrix(lstm_model, X_test, y_test)
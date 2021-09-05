# sentiment classification of US Airline tweets
import numpy as np
import pandas as pd
import os
import sys
# import tweepy as tw
# import re  # RegEx

from nltk.stem.porter import *
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from keras.preprocessing.sequence import skipgrams
from keras.layers import dot
from keras.layers.core import Reshape
from keras.models import Sequential
# from keras.layers.embeddings import Embedding

from keras import Model
from keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout
from keras.metrics import Precision, Recall

# import nltk  # natural language tool kit
# from nltk.corpus import stopwords
# nltk.download("stopwords")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# for TensorFlow to work without any interruptions,
# set this environment variable to disable all logging output


"""def get_text(twitter_data_import):  # get DataFrame with tweet_id and associated text
    tweet_text_data = pd.DataFrame(columns=["tweet_id", "text"])
    for index, tweet_id in enumerate(twitter_data_import.loc[:10, "tweet_id"]):
        try:  # check if id is working
            text = api.get_status(tweet_id).text
            print(tweet_id)
        except:  # if not set text to nan
            text = np.nan
            print(f"failed id:{tweet_id}")
        tweet_info = {"tweet_id": f"{tweet_id}", "text": text,
                      "sentiment": twitter_data_import.loc[index, "airline_sentiment"]}
        tweet_text_data = tweet_text_data.append(tweet_info, ignore_index=True)
    return tweet_text_data"""


def clean_tweet(tweet):
    # cleaning of raw tweet
    text = tweet.lower()                            # lower case
    text = re.sub(r"http\S+", "", text)             # text remove hyperlinks
    text = re.sub(r"@\S+", "", text)                # text remove @mentions
    # ##### EMOJIS
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)       # remove non letters
    # text = re.sub(r"^RT[\s]+", "", text)            # remove retweet text "RT"
    text = " ".join(text.split())  # for legibility, avoid having multiple spaces after another
    # words = text.split()
    # remove stopwords
    # words = [w for w in words if w not in stopwords.words("english")]
    # apply stemming
    # words = [PorterStemmer().stem(w) for w in words]
    # text = " ".join(words)
    # return list
    return text


def tokenize_and_pad(texts, vocab_size_, max_length_):
    # tokenizing
    tokenizer_new = Tokenizer(num_words=vocab_size_)
    tokenizer_new.fit_on_texts(texts)
    texts_t = tokenizer_new.texts_to_sequences(texts)
    # padding
    texts_tp = pad_sequences(texts_t, maxlen=max_length_, padding="post")
    # maxlen equal to longest sequence, ToDo: change? -> define truncating
    return texts_tp, tokenizer_new
    # vectorizer = TextVectorization(max_tokens=vocab_size, output_mode = 'int', output_sequence_length = max_length)
    # vectorizer.adapt(texts)
    # return texts.map(vectorizer), vectorizer

# ##################################################################################################
# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.


def generate_training_data(sequences, window_size, num_ns_, vocab_size_):
    print("Generating training data...")
    # Elements of each training example are appended to these lists.
    data_targets = np.empty((0, 1))
    data_contexts = np.empty((0, num_ns_+1))
    data_labels = np.empty((0, num_ns_+1))

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size_)

    # Iterate over all sequences (sentences) in dataset.
    seq_count = 1
    for sequence in sequences:
        sys.stdout.write(f"\rProcess sequence: {str(seq_count).zfill(5)}/{len(sequences)}")
        sys.stdout.flush()
        seq_count +=1
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size=vocab_size_,
                                                                           sampling_table=sampling_table,
                                                                           window_size=window_size, negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=context_class,
                                                                                         num_true=1, num_sampled=num_ns,
                                                                                         unique=True,
                                                                                         range_max=vocab_size_,
                                                                                         seed=SEED,
                                                                                         name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
            context = tf.concat([context_class, negative_sampling_candidates], 0).numpy().reshape(1, 5)
            label = tf.constant([1] + [0]*num_ns, dtype="int64").numpy()
    
            # Append each element from the training example to global lists.
            data_targets = np.vstack((data_targets, target_word))
            data_contexts = np.vstack((data_contexts, context))
            data_labels = np.vstack((data_labels, label))

    return data_targets, data_contexts, data_labels
# ###############################################################################################


def save_embeddings(embedding_matrix_import):
    import io
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')
    for index, word in enumerate(vocab):
        vec = embedding_matrix_import[index+1, :]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
    return None


if __name__ == "__main__":
    # Twitter developer keys and setup
    ###
    """consumer_key = "dBVOKYBYosXxTC9Z5B4BCilLm"
    consumer_key_secret = "gER2UBsOGJ8FtDGVfRPbXzuTskd9agbr3lNcuEPHT2iBYm641o"
    access_token = "1414641981570134020-UQC0Qk3fTowwDJLCqszONYnCSCwquj"
    access_token_secret = "XW3X5sWIx0V5RZ3FR2TA1XcP2AeqvhZCrdimPTcTusp6j"

    auth = tw.OAuthHandler(consumer_key, consumer_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tw.API(auth)
    ###"""

    # load tweets
    twitter_data = pd.read_csv(r"tweets_data\Tweets_airlines.csv")
    print(twitter_data.head())
    # get DataFrame with tweet_id and text
    # text_data = get_text(twitter_data)
    text_data = twitter_data.loc[:, ["text", "airline_sentiment"]]
    print(text_data.head())

    """# IMPORT DATA TWEETS: Airlines
    df_tweets_air_full = pd.read_csv('tweets_data/Tweets_airlines.csv')
    print(df_tweets_air_full.info())
    df_tweets_air = df_tweets_air_full.copy()
    df_tweets_air = df_tweets_air.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
    df_tweets_air['category'] = df_tweets_air['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
    df_tweets_air = df_tweets_air[['category', 'clean_text']]
    df_tweets_air = df_tweets_air.head(1)
    # IMPORT DATA TWEETS: General
    df_tweets_gen = pd.read_csv('tweets_data/Tweets_general.csv')
    df_tweets_gen = df_tweets_gen[['category', 'clean_text']]
    df_tweets_gen = df_tweets_gen.head(15000)
    # COMBINE DATASETS
    df_tweets = pd.concat([df_tweets_air, df_tweets_gen], ignore_index=True)
    # df_tweets = df_tweets_air.copy()
    df_tweets = df_tweets.sample(frac=1)  # shuffle dataframe"""



    print(text_data.isnull().sum())
    # drop rows/examples where either no text could be fetched or no sentiment was assigned
    text_data.dropna(axis=0, how="any", inplace=True)
    print(text_data.head())
    # One-hot encode sentiments
    y = pd.get_dummies(text_data["airline_sentiment"])
    # ##########################################
    # DATA VISUALIZATIONS
    # ##########################################
    # PREPROCESSING
    # # I) Clean tweets/sentences
    # print(text_data["text"][0])
    # print(tweet_to_words(text_data["text"][0]))
    text_data["clean_text"] = text_data["text"].map(clean_tweet)
    print(text_data.head())

    # # II) Tokenize and pad sequences to yield arrays of integers with uniform length,
    # each integer representing a unique word
    # ## define vocabulary size (n-1 most common words that are kept) and maximum length of an encoded sequence
    vocab_size = 5000  # len(np.unique(np.concatenate(text_data["clean_text"].apply(str.split))))
    print(f"vocabulary size = {vocab_size}")
    max_length = max(text_data["clean_text"].apply(str.split).apply(len))
    # length of all sequences = length longest sequence
    print(f"sequence length = {max_length}")
    X_tp, tokenizer = tokenize_and_pad(text_data["clean_text"], vocab_size, max_length)
    print(X_tp[3])
    print(X_tp[3].shape)
    print(tokenizer.sequences_to_texts([X_tp[3]]))
    # save vocabulary
    vocab = tokenizer.sequences_to_texts([range(0, vocab_size+1)])[0].split()
    # print(vocab)

    # # III) Split data set into train and test set ###
    # ToDo: 1) set shuffle to True, 2) use of validation set (kaggle), 3) maybe shuffle rest aswell for visualizations
    X_train, X_test, y_train, y_test = train_test_split(X_tp, y, test_size=0.1, shuffle=False)

    # WORD EMBEDDINGS (Word2Vec)
    # # Skip-Gram and Negative-Sampling
    pos_skip_gram, _ = skipgrams(X_tp[3], vocabulary_size=vocab_size, window_size=2, negative_samples=0)
    # ToDo: Def vocabulary size
    # "window_size = n" means 2*n + 1 words are in the whole window 
    # "negative_samples = 0" means that all context words are in the same window
    # as the sampled word/target word (no random samples for context words)
    print(tokenizer.sequences_to_texts(pos_skip_gram[:5]))
    ##############################################################################
    num_ns = 4
    SEED = 42
    targets, contexts, labels = generate_training_data(sequences=X_train[:, :],
                                                       window_size=2,
                                                       num_ns_=num_ns,
                                                       vocab_size_=vocab_size)
    print(targets[0])
    print(contexts[0])
    print(labels[0])

    # model
    embed_size = 64
    target_model = Sequential() 
    target_model.add(Embedding(vocab_size, embed_size, embeddings_initializer="glorot_uniform", input_length=1)) 
    target_model.add(Reshape((embed_size, ))) 

    context_model = Sequential() 
    context_model.add(Embedding(vocab_size, embed_size,
                                embeddings_initializer="glorot_uniform",
                                input_length=num_ns + 1))
    context_model.add(Reshape((num_ns + 1, embed_size)))

    dot_product = dot([target_model.output, context_model.output], axes=(1, 2), normalize=False)

    model = Model(inputs=[target_model.input, context_model.input], outputs=dot_product)
    # compile model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer="adam", metrics="accuracy")
    print(model.summary())

    # fit model to Skip-Grams (target words and contexts)
    model.fit(x=[targets, contexts], y=labels, epochs=20, verbose=1)
    
    embedding_matrix = target_model.get_weights()[0]  # lookup table for embeddings
    print(embedding_matrix.shape)
    print(embedding_matrix[0:5])

    # save vectors and words in .tsv files for later use and visualizations
    save_embeddings(embedding_matrix)

    # LSTM MODEL
    model = Sequential()
    # train embeddings "on the go"
    # model.add(Embedding(vocab_size, 128, input_length = max_length))
    # pre trained embeddings (set trainable to True to let the model readjust pretrained embeddings)
    model.add(Embedding(vocab_size, embed_size, weights=[embedding_matrix], input_length=max_length, trainable=False))
    model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.4))
    model.add(Dense(3, activation='softmax'))
    print(model.summary())
    # Compile model
    model.compile(loss='CategoricalCrossentropy', optimizer="adam", metrics=['accuracy', Precision(), Recall()])
    # Train model
    model.fit(X_train, y_train, validation_split=0.2, batch_size=64, epochs=20, verbose=1)

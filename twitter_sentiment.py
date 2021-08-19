# FINAL MODEL for Twitter tweet text sentiment analysis
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

import pickle
import pandas as pd
import datetime

from nltk.stem.porter import *

# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer

import keras.backend as k
# from keras.preprocessing.sequence import skipgrams
from keras.layers.core import Reshape
from keras.layers import dot
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Model
from keras.metrics import Precision, Recall
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.layers import Bidirectional, LSTM, Dense  # , Dropout

import tensorflow as tf
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

from plotting_framework import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# for TensorFlow to work without any interruptions,
# set this environment variable to disable all logging output

workdir = os.path.dirname(__file__)
sys.path.append(workdir)  # append path of project folder directory

# DEFINE MODEL CHARACTERISTICS
vocabulary_size = 5000  # TODO HYPER PARAMETER
embedding_size = 32     # TODO HYPER PARAMETER
epochs = 30             # TODO HYPER PARAMETER
learning_rate = 0.001   # TODO HYPER PARAMETER
momentum = 0.0          # TODO HYPER PARAMETER
batch_size = 128         # TODO HYPER PARAMETER

num_ns = 4      # number of negative samples for embedding
SEED = 42


def tweet_to_words(tweet):

    text = tweet.lower()                            # lower case
    # text = emoji.demojize(text)                   # translate emojis TODO emoji replacement
    text = re.sub(r"http\S+", "", text)             # text remove hyperlinks
    text = re.sub(r"#", "", text)                   # text remove hashtag symbol
    text = re.sub(r"@\S+", "", text)                # text remove @mentions
    text = re.sub(r"'", '', text)                   # remove apostrophes
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)       # remove non letters
    text = re.sub(r"^RT[\s]+", "", text)            # remove retweet text "RT"

    # words = " ".join(text.split())  # for legibility, avoid having multiple spaces after another
    words = text.split()

    return words


def tokenize_pad_sequences(text):

    # This function tokenize the input text into sequences of integers
    # and then pads each sequence to the same length

    # Text tokenization with max amount of vocab_size word token ids
    tokenizer_padseq = Tokenizer(num_words=vocabulary_size, lower=True, split=' ')
    tokenizer_padseq.fit_on_texts(text)
    # Transforms text to a sequence of integers
    output = tokenizer_padseq.texts_to_sequences(text)
    # Pad sequences to the same length
    output = pad_sequences(output, padding='post', maxlen=max_len, truncating='post')

    return output, tokenizer_padseq


def generate_training_data(sequences, window_size, num_ns_, vocab_size_):
    print("[VECTORIZATION] Generating training data ...")
    # Elements of each training example are appended to these lists.
    data_targets = np.empty((0, 1))
    data_contexts = np.empty((0, num_ns_ + 1))
    data_labels = np.empty((0, num_ns_ + 1))

    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size_)

    # Iterate over all sequences (sentences) in dataset.
    seq_count = 1
    for sequence in sequences:
        sys.stdout.write(f"\rProcess sequence: {str(seq_count).zfill(5)}/{len(sequences)}")
        sys.stdout.flush()
        seq_count += 1
        # Generate positive skip-gram pairs for a sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(sequence, vocabulary_size=vocab_size_,
                                                                           sampling_table=sampling_table,
                                                                           window_size=window_size, negative_samples=0)

        # Iterate over each positive skip-gram pair to produce training examples
        # with positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
            neg_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=context_class,
                                                                                    num_true=1, num_sampled=num_ns_,
                                                                                    unique=True,
                                                                                    range_max=vocab_size_,
                                                                                    seed=SEED,
                                                                                    name="negative_sampling")

            # Build context and label vectors (for one target word)
            neg_sampling_candidates = tf.expand_dims(neg_sampling_candidates, 1)
            context = tf.concat([context_class, neg_sampling_candidates], 0).numpy().reshape(1, 5)
            label = tf.constant([1] + [0] * num_ns, dtype="int64").numpy()

            # Append each element from the training example to global lists.
            data_targets = np.vstack((data_targets, target_word))
            data_contexts = np.vstack((data_contexts, context))
            data_labels = np.vstack((data_labels, label))

    return data_targets, data_contexts, data_labels


def save_embeddings(embedding_matrix_import):
    import io
    out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('metadata.tsv', 'w', encoding='utf-8')
    for index, word in enumerate(vocab):
        vec = embedding_matrix_import[index + 1, :]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
    return None


def f1_score(precision_val, recall_val):
    f1_val = 2 * (precision_val * recall_val) / (precision_val + recall_val + k.epsilon())
    return f1_val


def predict_class(text):
    # Function to predict sentiment class of the passed text

    with open(r'model_data\tokenizer_save.pickle', 'rb') as handle_import:
        tokenizer_import = pickle.load(handle_import)  # load tokenizer

    sentiment_classes = ['Negative', 'Neutral', 'Positive']

    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer_import.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('SENTIMENT prediction: ', sentiment_classes[yt[0]])


print("\n_______DAML_Twitter_Sentiment________\n")


# IMPORT DATA TWEETS: Airlines
df_tweets_air_full = pd.read_csv('tweets_data/Tweets_airlines.csv')
print(df_tweets_air_full.info())
df_tweets_air = df_tweets_air_full.copy()
df_tweets_air = df_tweets_air.rename(columns={'text': 'clean_text', 'airline_sentiment': 'category'})
df_tweets_air['category'] = df_tweets_air['category'].map({'negative': -1.0, 'neutral': 0.0, 'positive': 1.0})
df_tweets_air = df_tweets_air[['category', 'clean_text']]
# df_tweets_air = df_tweets_air.head(10000)  # TODO custom
# IMPORT DATA TWEETS: General
df_tweets_gen = pd.read_csv('tweets_data/Tweets_general.csv')
df_tweets_gen = df_tweets_gen[['category', 'clean_text']]
df_tweets_gen = df_tweets_gen.head(15000)  # TODO custom
# COMBINE DATASETS
df_tweets = pd.concat([df_tweets_air, df_tweets_gen], ignore_index=True)
# df_tweets = df_tweets_air.copy()
df_tweets = df_tweets.sample(frac=1)  # shuffle dataframe

# df_tweets = df_tweets.head(20000)  # TODO select to 10k rows (~10%)

df_tweets.isnull().sum()                # Check for missing data
df_tweets.dropna(axis=0, inplace=True)  # Drop missing rows
print(df_tweets.head(10))  # output first ten tweet df entries

# Apply data processing to each tweet
X_tweets_list = list(map(tweet_to_words, df_tweets['clean_text']))

# DYNAMICALLY change the max_len parameter
max_len = max([len(x) for x in X_tweets_list])
print(f"\nMax number of words expected"
      f" in a processed tweet: {max_len} \n")

# TOKENIZE and PAD the list of word arrays
X_tweets_list, tokenizer = tokenize_pad_sequences(df_tweets['clean_text'])

with open(r'model_data\tokenizer_save.pickle', 'wb') as handle:  # save tokenizer
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

vocab = tokenizer.sequences_to_texts([range(0, vocabulary_size+1)])[0].split()

# TARGET vector ONE HOT ENCODING (3dummy variables)
y_categories = pd.get_dummies(df_tweets['category'])

# TRAIN VALIDATION SPLIT (60% train, 20% valid, 20% test)
X_train, X_test, y_train, y_test = \
    train_test_split(X_tweets_list, y_categories, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = \
    train_test_split(X_train, y_train, test_size=0.25, random_state=1)
print(f"SHAPE X_train: {X_train.shape}")
print(f"SHAPE X_valid: {X_val.shape}")
print(f"SHAPE X_test : {X_test.shape}\n")
print(f"SHAPE Y_train: {y_train.shape}")
print(f"SHAPE Y_valid: {y_val.shape}")
print(f"SHAPE Y_test : {y_test.shape}\n")

"""# WORD EMBEDDINGS (Word2Vec)
# # Skip-Gram and Negative-Sampling
pos_skip_gram, _ = skipgrams(X_tp[3], vocabulary_size=vocabulary_size, window_size=2, negative_samples=0)
# ToDo: Def vocabulary size
# "window_size = n" means 2*n + 1 words are in the whole window 
# "negative_samples = 0" means that all context words are in the same window
# as the sampled word/target word (no random samples for context words)
print(tokenizer.sequences_to_texts(pos_skip_gram[:5]))
"""


targets, contexts, labels = generate_training_data(sequences=X_train[:, :], window_size=2, num_ns_=num_ns,
                                                   vocab_size_=vocabulary_size)
target_model = Sequential()
target_model.add(Embedding(vocabulary_size, embedding_size, embeddings_initializer="glorot_uniform",
                           input_length=1))
target_model.add(Reshape((embedding_size,)))

context_model = Sequential()
context_model.add(Embedding(vocabulary_size, embedding_size, embeddings_initializer="glorot_uniform",
                            input_length=num_ns + 1))
context_model.add(Reshape((num_ns + 1, embedding_size)))

dot_product = dot([target_model.output, context_model.output], axes=(1, 2), normalize=False)

model_embed = Model(inputs=[target_model.input, context_model.input], outputs=dot_product)
# compile embeddings training model
model_embed.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                    optimizer="adam", metrics="accuracy")
print("\nEMBEDDING MODEL TRAINING:")
print(model_embed.summary())
# fit model to Skip-Grams (target words and contexts)
model_embed.fit(x=[targets, contexts], y=labels, epochs=20, verbose=1, validation_split = 0.2)

embedding_matrix = target_model.get_weights()[0]  # lookup table for embeddings # TODO? Warum von target
print(embedding_matrix.shape)
print(embedding_matrix[0:5])
# save vectors and words in .tsv files for later use and visualizations
save_embeddings(embedding_matrix)


# OPTIMIZERS
"""sgd = SGD(lr=learning_rate, momentum=momentum,
          decay=(learning_rate/epochs), nesterov=False)

rmsprop = RMSprop(learning_rate=learning_rate, rho=0.9, momentum=momentum,
                  epsilon=1e-07, centered=True)"""

adam = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)

# BUILD KERAS MODEL PIPELINE
model = Sequential()
model.add(Embedding(vocabulary_size, embedding_size, input_length=max_len,
                    trainable=True))
# model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
# model.add(MaxPooling1D(pool_size=2))
model.add(Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)))  #
# model.add(Dropout(0.4))
model.add(Dense(3, activation='softmax'))

# tf.keras.utils.plot_model(model, show_shapes=True)
print(model.summary())  # OUTPUT model information

model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy', Precision(), Recall()])

# AUTOMATIC RESTORATION of optimal model configuration AFTER training completed
# RESTORE the OPTIMAL NN WEIGHTS from when val_loss was minimal (epoch nr.)
# SAVE model weights at the end of every epoch, if these are the best so far
time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
checkpoint_filepath = rf'model_data_custom_embedding\best_model_testrun_{time_of_analysis}.hdf5'
model_checkpoint_callback = tf.keras.callbacks.\
    ModelCheckpoint(filepath=checkpoint_filepath, patience=3, verbose=1,
                    save_best_only=True, monitor='val_loss', mode='min')

# apply model to training data and store history information
history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                    batch_size=batch_size, epochs=epochs, verbose=1,
                    callbacks=[model_checkpoint_callback], weights=[embedding_matrix])

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

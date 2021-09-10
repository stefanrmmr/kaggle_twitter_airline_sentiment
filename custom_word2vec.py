# CUSTOM WORD2VEC WORD EMBEDDINGS
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

# In this file an approach to train custom word2vec embeddings is shown using the same data
# as in the LSTM model to be able to compare the performance to the embedding-layer-approach
# used in the LSTM model. For comparison don't shuffle the data, use a seed in generate_skip_grams
# and perform the same train-test split as in the LSTM model

# imports
# GENERAL
import sys   # for displaying a progress-count
import io    # for saving output
import numpy as np
import pandas as pd
# for PREPROCESSING
from preprocessing import *
from sklearn.model_selection import train_test_split
# for WORD2VEC EMBEDDINGS
import tensorflow as tf
from keras.preprocessing.sequence import skipgrams
from keras import Model
from keras.layers import dot
from keras.layers.core import Dense, Reshape
from keras.layers.embeddings import Embedding
from keras.models import Sequential

# for LSTM model
from LSTM_model import *

#for PLOTTING
from plotting_framework import *

def generate_skipgrams(sequences, window_size, num_ns, vocab_size): # !!! performance heavy !!!
    # abstract:
    # Generates skip-gram pairs joined with num_ns negative samples for a list of sequences
    # (int-encoded sentences) based on window size and vocabulary size
    # concrete:
    # for each word/token selected as target add one word in the context of the target (in window_size)
    # and fill the context with num_ns "wrong" context words (not in window_size of target)

    print("Generating positive and negative Skip-Grams ...")
    # Elements of each training example are appended to these numpy arrays
    targets = np.empty((0,1))
    contexts = np.empty((0, num_ns+1))
    labels = np.empty((0, num_ns+1))

    # Build the sampling table for vocab_size tokens to less frequenly sample words/tokens
    # accuring with higher frequency in the data
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

    # Iterate over all sequences/sentences in the dataset.
    seq_count = 1 # for displaying progress
    for sequence in sequences:
        sys.stdout.write(f"\rProcess sequence: {str(seq_count).zfill(6)}/{len(sequences)}")
        sys.stdout.flush()
        seq_count += 1
        #1) Generate positive skip-gram pairs for a sequence/sentence:
        ## Select words as target words (randomly with the sampling table) and create Skip-Gram pairs
        ## by joining the target word with a context word in the window_size of the target word
        ## [target word, context_word (word in window_size)]
        positive_skip_grams, _ = skipgrams(sequence, vocabulary_size = vocab_size\
            , sampling_table = sampling_table, window_size = window_size, negative_samples = 0)
        # "negative_samples = 0" means that all context words are in the same window as the sampled word/target word
        # negative samples are drawn in step 2):
        #2) Iterate over each positive skip-gram pair to produce training examples
        # with one positive/"true" context word and num_ns negative samples/"false" context words
        ## [target word]; [positive context, num_ns times negative samples]
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant([context_word], dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(true_classes=context_class, num_true=1\
                , num_sampled=num_ns, unique=True, range_max=vocab_size, name="negative_sampling")

            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates, 1)
            context = tf.concat([context_class, negative_sampling_candidates], 0).numpy().reshape(1,5)
            label = tf.constant([1] + [0]*num_ns, dtype="int64").numpy()
    
            # Append each element from the training example to training example - arrays
            targets = np.vstack((targets, target_word)) # [target]
            contexts = np.vstack((contexts, context))   # [positive context, num_ns times negative samples]
            labels = np.vstack((labels, label))         # [1, num_ns times 0]
    return targets.astype("int64"), contexts.astype("int64"), labels.astype("int64")

def print_example_processing(df_tweets, X_tp, tokenizer):
    print("......................................................")
    print("EXAMPLE PREPROCESSING:\n")
    print("raw tweet:")
    print(df_tweets.loc[16, "text"])
    print("\nclean tweet:")
    print(df_tweets.loc[16, "clean_text"])
    print("\ntokenized and padded tweet:")
    print(X_tp[16])
    pos_skip_gram, _ = skipgrams(X_tp[16], vocabulary_size = vocab_size, window_size = 2, negative_samples = 0)
    # (no random samples for context words)
    print("\nFirst positive Skip-Grams:")
    print(tokenizer.sequences_to_texts(pos_skip_gram[:5]))
    print(f"\n# training examples: {len(pos_skip_gram)}")
    print("Final training examples are sampled randomly with sampling_table and filled with negative samples")
    print("......................................................\n")
    return None

def save_embeddings(embedding_matrix, vocab):
    # Saves embeddings and respective vocabulary in .tsv files
    out_v = io.open('model_data_keras_embedding/vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('model_data_keras_embedding/metadata.tsv', 'w', encoding='utf-8')
    for index, word in enumerate(vocab):
        vec = embedding_matrix[index+1,:]
        out_v.write('\t'.join([str(x) for x in vec]) + "\n")
        out_m.write(word + "\n")
    out_v.close()
    out_m.close()
    return None

if __name__ == "__main__":

    print("\n_______DAML_Twitter_Sentiment________\n")

    # load tweets
    df_tweets, y = prepare_dataframe(airline = 1, shuffle = False)
    # Examples in airline data: 14640
    # airline, filler_data from [0, 1] determine percentage of airline data
    # Set shuffle to False to reproduce outcomes
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
    
    ## III) Split data set into train and test set
    ### Again, to be able to compare the performance of the custom embeddings to the used LSTM
    X_train, X_test, y_train, y_test = train_test_split(X_tp, y, test_size = 0.2, shuffle = False)

    # WORD EMBEDDINGS (Word2Vec)
    ## Skip-Gram and Negative-Sampling
    ## print an example
    print_example_processing(df_tweets, X_tp, tokenizer)
    num_ns = 4
    #print(f"shape of training data: {X_train.shape}")
    targets, contexts, labels = generate_skipgrams(sequences = X_tp[:, :], window_size = 2, num_ns = num_ns, vocab_size = vocab_size)
    # "window_size = n" means 2*n + 1 words are in the whole window
    print("\n\nExample training data:")
    print(f"target:  {targets[0]}")
    print(f"context: {contexts[0]}; Number of negative samples: {num_ns}")
    print(f"label:   {labels[0]}\n")

    ## MODEL
    embed_size = 128
    print(f"Dimension of embeddings: {embed_size}")
    target_model = Sequential() 
    target_model.add(Embedding(vocab_size, embed_size, input_length = 1, name = "target_embedding")) 
    target_model.add(Reshape((embed_size, ))) 

    context_model = Sequential() 
    context_model.add(Embedding(vocab_size, embed_size, input_length = num_ns + 1, name = "context_embedding"))
    context_model.add(Reshape((num_ns + 1, embed_size)))

    dot_product = dot([target_model.output, context_model.output], axes = (1,2), normalize = False)

    model = Model(inputs=[target_model.input, context_model.input], outputs = dot_product)
    model._name = "Word2Vec"
    # compile model
    model.compile(loss = tf.keras.losses.CategoricalCrossentropy(from_logits = True), optimizer = "adam", metrics = "accuracy")
    
    #tf.keras.utils.plot_model(model, show_shapes=True)
    print(model.summary())

    # fit model to Skip-Grams (target words and contexts) with negative samples as described above
    model.fit(x = [targets, contexts], y = labels, epochs = 20, verbose = 1)
    
    embedding_matrix = target_model.get_weights()[0] # lookup table for embeddings
    print(f"\nShape of embedding matrix (vocab_size, embed_size):\n{embedding_matrix.shape}")
    print(f"Example embedding vector:\n{embedding_matrix[0]}")

    # save embedding-vectors and words in .tsv files for later use and visualizations
    save_embeddings(embedding_matrix, vocab)

    ###################################################################################
    ###################################################################################
    # Analysis of performance in LSTM

    # define model characteristics
    epochs = 20              # TODO HYPER PARAMETER
    batch_size = 64          # TODO HYPER PARAMETER


    # PRETRAINED EMBEDDING WEIGHTS

    # AUTOMATIC RESTORATION of optimal model configuration AFTER training completed
    # RESTORE the OPTIMAL NN WEIGHTS from when val_loss was minimal (epoch nr.)
    # SAVE model weights at the end of every epoch, if these are the best so far
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    checkpoint_filepath = rf'model_data_keras_embedding\best_pretrained_model_testrun_{time_of_analysis}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.\
        ModelCheckpoint(filepath=checkpoint_filepath, patience=3, verbose=1,
                        save_best_only=True, monitor='val_loss', mode='min')
    
    # build model
    custom_embed_lstm_model = build_LSTM(vocab_size, max_length, embed_size, embedding_matrix, trainable = False)
    # fit model
    custom_embed_history = custom_embed_lstm_model.fit(X_train, y_train, batch_size = batch_size\
        , epochs = 20, validation_split = 0.2, verbose = 1, callbacks = [model_checkpoint_callback])
    
    # PLOT ACCURACY and LOSS evolution
    plot_training_hist(custom_embed_history, epochs)
    
    # The model weights (that are considered the best) are loaded into the model.
    custom_embed_lstm_model = load_model(checkpoint_filepath)

    # Evaluate model on the TEST SET  METRICS
    loss, accuracy, precision, recall = custom_embed_lstm_model.evaluate(X_test, y_test, verbose=0)
    print("\nPretrained Embeddings:")
    print("___________________________________________________")
    print('TEST Dataset Loss      : {:.4f}'.format(loss))
    print('TEST Dataset Accuracy  : {:.4f}'.format(accuracy))
    print('TEST Dataset Precision : {:.4f}'.format(precision))
    print('TEST Dataset Recall    : {:.4f}'.format(recall))
    print('TEST Dataset F1 Score  : {:.4f}'.format(f1_score(precision, recall)))
    print("===================================================")
    #######################################################################################

    '''#TRAINABLE WEIGHTS

    # AUTOMATIC RESTORATION of optimal model configuration AFTER training completed
    # RESTORE the OPTIMAL NN WEIGHTS from when val_loss was minimal (epoch nr.)
    # SAVE model weights at the end of every epoch, if these are the best so far
    time_of_analysis = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    checkpoint_filepath = rf'model_data_keras_embedding\best_trainable_model_testrun_{time_of_analysis}.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.\
        ModelCheckpoint(filepath=checkpoint_filepath, patience=3, verbose=1,
                        save_best_only=True, monitor='val_loss', mode='min')

    # build model
    trainable_embed_lstm_model = build_LSTM(vocab_size, max_length, embed_size, embedding_matrix = 0, trainable = True)
    # fit model
    trainable_embed_history = trainable_embed_lstm_model.fit(X_train, y_train, batch_size = batch_size\
        , epochs = 20, validation_split = 0.2, verbose = 1, callbacks = [model_checkpoint_callback])
    # PLOT ACCURACY and LOSS evolution
    plot_training_hist(trainable_embed_history, epochs)

    # The model weights (that are considered the best) are loaded into the model.
    trainable_embed_lstm_model = load_model(checkpoint_filepath)

    # Evaluate model on the TEST SET  METRICS
    loss, accuracy, precision, recall = trainable_embed_lstm_model.evaluate(X_test, y_test, verbose=0)
    print("\nTrainable Embeddings:")
    print("___________________________________________________")
    print('TEST Dataset Loss      : {:.4f}'.format(loss))
    print('TEST Dataset Accuracy  : {:.4f}'.format(accuracy))
    print('TEST Dataset Precision : {:.4f}'.format(precision))
    print('TEST Dataset Recall    : {:.4f}'.format(recall))
    print('TEST Dataset F1 Score  : {:.4f}'.format(f1_score(precision, recall)))
    print("===================================================")'''
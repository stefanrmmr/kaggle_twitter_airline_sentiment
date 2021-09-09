# EMBEDDING ANALYSIS
# Adrian Brünger, Stefan Rummer, TUM, summer 2021

# In this file a PCA is used to visualize embeddings

# imports
# GENERAL
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
# for TEXT PROCESSING
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
# for DIMENSION REDUCTION
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# plotting -> later in plotting framework
from plotting_framework import *

def embedding_distance(embedding_matrix, label_array):
    # return distance ranking for each word with the word itself at the top
    # compute pairwise distances of embedding vectors
    distance_matrix = pairwise_distances(embedding_matrix)
    # sort for each embedding vector/word
    distance_ranking = label_array[np.argsort(distance_matrix).T]
    # put in DataFrame (sorting and transposing results to the first row being the word itself
    # in each case (since the distance of a word to itself is 0) -> header)
    distance_ranking = pd.DataFrame(distance_ranking, columns = distance_ranking[0,:])
    return distance_ranking

def get_indices(word_dict, distance_ranking, custom_embed_labels):
    indices_list = []
    for key in word_dict.keys():
        # filter stopwords
        words = [w for w in distance_ranking[key] if w not in stopwords.words("english")]
        print(f"\nAnalyzed word: {key}")
        print(f"Nearby words: {words[1:word_dict[key]]}")
        # get indices
        indices = [np.where(custom_embed_labels == label)[0][0] for label in words]
        indices = indices[:word_dict[key]]
        # append list
        indices_list.append(indices)
    return indices_list

if __name__ == "__main__":
    # load embeddings and vocabulary
    custom_embed = np.genfromtxt("model_data_keras_embedding/vectors_air_500_256dim.tsv", delimiter = "\t")
    custom_embed_labels = np.genfromtxt("model_data_keras_embedding/metadata_air_500_256dim.tsv", delimiter = "\t", dtype = str)
    #vocab = dict(enumerate(custom_embed_labels))

    #1) visualize embeddings

    # calculate euclidean distances between all words
    distance_ranking = embedding_distance(custom_embed,  custom_embed_labels)
    print(distance_ranking["plane"])
    #print(custom_embed_labels[distance_ranking][:10])

    # get examples for plotting
    indices_list = get_indices({"plane" : 5, "amazing" : 3, "hrs" : 3}, distance_ranking, custom_embed_labels)

    # reduce dimensions of embeddings for plotting
    pca = PCA(n_components = 2)
    custom_embed_2d = pca.fit_transform(custom_embed)
    print(f"variance explained by the first 2 PCs: {round(sum(pca.explained_variance_ratio_)*100, 2)}%")
    # plot
    plot_embeddings(custom_embed_2d, custom_embed_labels, indices_list)


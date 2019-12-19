from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


import numpy as np
import pandas as pd

import hdbscan
import time

from consensus_clustering import ConsensusClustering
from process_data import load_aml8k, load_panceras_mouse, load_panceras_human


def find_hyper_parameters(X, y, max_cluster_size, interval):
    result_list = []
    for min_cluster_size in range(2, max_cluster_size, interval):

        start_time = time.time()

        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)    
        # metric = 'cosine')
        labels = clustering.fit_predict(X)

        predicted_labels = labels
        true_labels = y.flatten()
        rand_score = adjusted_rand_score(true_labels, predicted_labels)
        num_clusters = len(set(predicted_labels)) - 1
        result_list.append((rand_score, min_cluster_size, num_clusters))

        end_time = time.time()

    highest_rand_score, best_cluster_size, num_clusters = max(result_list, key=lambda x: x[0])

    print("############       Normal clustering results      ###########")
    print("Time taken for finding best hyperparameter: {}".format(end_time - start_time))
    print("Best cluster size: {}".format(best_cluster_size))
    print("Adj. rand score: {}".format(highest_rand_score))
    print("Number of clusters: {}".format(num_clusters))
    print("#############################################################\n \n")

    return best_cluster_size


def hdbscan_consensus_matrix(consensus_matrix, max_cluster_size, interval):

    distance_matrix = 1 - consensus_matrix
    result_list = []

    for min_cluster_size in range(2, max_cluster_size, interval):
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)

        predicted_labels = labels
        true_labels = y.flatten()
        rand_score = adjusted_rand_score(true_labels, predicted_labels)
        result_list.append((rand_score, min_cluster_size,
                            len(set(predicted_labels)) - 1))

    highest_rand_score, best_cluster_size, num_clusters = max(result_list, key = lambda x: x[0])

    print("##########       Consensus clustering results       #########")
    print("Best cluster size: {}".format(best_cluster_size))
    print("Adj. rand score: {}".format(highest_rand_score))
    print("Number of clusters: {}".format(num_clusters))
    print("#############################################################\n\n")


PATH = r"C:\Users\Jakob\Documents\consensusClustering\experiment\datasets"
FUNCTION_NAMES = [load_panceras_human]#, load_aml8k]
DATASET_NAME = ["load_panceras_human"]#, "AML8K"]

start = time.time()
for load_data, dataset_name in zip(FUNCTION_NAMES, DATASET_NAME):

    print("######## Loading Dataset: {} ###############".format(dataset_name))
    start1 = time.time()
    X, y = load_data(PATH)
    n_rows, n_columns = X.shape
    print("Data loaded successfuly, time taken: {}\n".format(time.time() - start1))

    print("Normalizing dataset")
    start1 = time.time()
    norm_data = normalize(X, norm='l2')
    print("Data normalized successfuly, time taken: {}\n".format(time.time() - start1))

    print("Calculating distance matrix and applying PCA")
    start1 = time.time()

    dist_matrix = pairwise_distances(norm_data, metric='cosine')
    n_components = int(n_rows * 0.08) #0.05 is variable suggested in article 0.04 -0.07

    sklearn_pca = sklearnPCA(n_components=n_components)
    distance_matrix_PCA = sklearn_pca.fit_transform(dist_matrix)
    
    norm_data = distance_matrix_PCA
    print("Reduced distance matrix shape: {}".format(norm_data.shape))
    print("Finished successfuly, time taken: {}\n".format(time.time() - start1))



    # print("Started clustering")
    # start1 = time.time()
    # clustering = hdbscan.HDBSCAN(min_cluster_size=500)    
    # labels = clustering.fit_predict(X)
    # predicted_labels = labels
    # true_labels = y.flatten()
    # rand_score = adjusted_rand_score(true_labels, predicted_labels)
    # print(rand_score)
    # print("Clustering was successful, time taken: {}\n".format(time.time() - start1))

    best_cluster_size = find_hyper_parameters(norm_data, y, 40, 1)

    # use best hyperparameters from single run test
    clustering_algorithm = hdbscan.HDBSCAN(min_cluster_size=best_cluster_size)

    # clustering_algorithm = DBSCAN(min_samples = 3, eps = 0.3, metric = 'cosine')
    consensus_clustering = ConsensusClustering(clustering_algorithm, 1)
    consensus_matrix = consensus_clustering.cc_fit(norm_data, y)

    # file_name = dataset_name + "_matrix500_mouse_PCA"
    # np.save(file_name, consensus_matrix)

    hdbscan_consensus_matrix(consensus_matrix, 70, 1)

    print("\n\n")

end = time.time()
print("Time taken: {}".format(end - start))

from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from random import randint

from scipy.spatial.distance import pdist, squareform
from fastcluster import linkage

import numpy as np

import hdbscan
import time

import matplotlib.pyplot as plt


def bootstrap(X, sample_size):
    
    n_samples = X.shape[0]
    
    rows_list = []
    for sample in range(sample_size):
        row_num = randint(0, n_samples - 1)
        rows_list.append(row_num)
        
    resampled_ds = X[rows_list]     
        
    return resampled_ds, rows_list


def conn_matrix(predicted_labels, rows_list, dict_original2resampled_idx, n_samples):
    connectivity_matrix = np.zeros((n_samples, n_samples))

    for combo in combinations(rows_list, 2):
        if combo[0] != combo[1]:

            if predicted_labels[dict_original2resampled_idx[combo[0]]] == predicted_labels[dict_original2resampled_idx[combo[1]]]:
                
                if combo[0] > combo[1]: 
                    connectivity_matrix[combo[0], combo[1]] = 1
                else:
                    connectivity_matrix[combo[1], combo[0]] = 1
        
    return connectivity_matrix


def indi_matrix(rows_list, n_samples):
    indicator_matrix = np.full((n_samples, n_samples), 1e-8)

    for combo in combinations(rows_list, 2):
        if combo[0] != combo[1]:
            
            if combo[0] > combo[1]:
                indicator_matrix[combo[0], combo[1]] = 1
            else:
                indicator_matrix[combo[1], combo[0]] = 1

    return indicator_matrix



def find_hyper_parameters(X, y, max_cluster_size, interval):
    result_list = []
    timing = []
    for min_cluster_size in range(2, max_cluster_size, interval):

        start_time = time.time()

        distance_matrix = pairwise_distances(X, metric='cosine')
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     metric='precomputed')

        predicted_labels = clustering.fit_predict(distance_matrix)
       # clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)    
        # metric = 'cosine')
        #predicted_labels = clustering.fit_predict(X)

        
        true_labels = y.flatten()
        rand_score = adjusted_rand_score(true_labels, predicted_labels)
        num_clusters = len(set(predicted_labels)) - 1
        result_list.append((rand_score, min_cluster_size, num_clusters))

        end_time = time.time()
        timing.append(end_time - start_time)
    
    highest_rand_score, best_cluster_size, num_clusters = max(result_list, key=lambda x: x[0])

    print("############       Normal clustering results      ###########")
    print("Avg. time for one iteration: {}".format(np.average(timing)))
    print("Best cluster size: {}".format(best_cluster_size))
    print("Adj. rand score: {}".format(highest_rand_score))
    print("Number of clusters: {}".format(num_clusters))
    print("#############################################################\n \n")

    return best_cluster_size


def hierarchical_consensus_matrix(consensus_matrix, y):

    distance_matrix = 1 - consensus_matrix
    result_list = []

    for distance_tresh in np.arange(0.01, 0.9, 0.05):
        clustering = AgglomerativeClustering(n_clusters=None, 
                                             distance_threshold=distance_tresh, 
                                             affinity='precomputed',
                                             linkage='complete')
        predicted_labels = clustering.fit_predict(distance_matrix)

        rand_score = adjusted_rand_score(y, predicted_labels)
        result_list.append((rand_score, distance_tresh,
                            len(set(predicted_labels)) - 1))

        highest_rand_score, best_cluster_size, num_clusters = max(result_list, key=lambda x: x[0])

    print("##########       Consensus clustering results       #########")
    print("Best cluster size: {}".format(best_cluster_size))
    print("Adj. rand score: {}".format(highest_rand_score))
    print("Number of clusters: {}".format(num_clusters))
    print("#############################################################\n\n")

    clustering = AgglomerativeClustering(n_clusters=None, 
                                             distance_threshold=best_cluster_size, 
                                             affinity='precomputed',
                                             linkage='complete')
    predicted_labels = clustering.fit_predict(distance_matrix)

    return predicted_labels





def hdbscan_consensus_matrix(consensus_matrix, max_cluster_size, interval, y):

    distance_matrix = 1 - consensus_matrix
    result_list = []

    y = y.flatten()

    for min_cluster_size in range(2, max_cluster_size, interval):
        clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                     metric='precomputed')
        predicted_labels = clustering.fit_predict(distance_matrix)

        rand_score = adjusted_rand_score(y, predicted_labels)
        result_list.append((rand_score, min_cluster_size,
                            len(set(predicted_labels)) - 1))

    highest_rand_score, best_cluster_size, num_clusters = max(result_list, key=lambda x: x[0])

    print("##########       Consensus clustering results       #########")
    print("Best cluster size: {}".format(best_cluster_size))
    print("Adj. rand score: {}".format(highest_rand_score))
    print("Number of clusters: {}".format(num_clusters))
    print("#############################################################\n\n")

    clustering = hdbscan.HDBSCAN(min_cluster_size=best_cluster_size,
                                     metric='precomputed')
    predicted_labels = clustering.fit_predict(distance_matrix)

    return predicted_labels


def consensus_matrix_histogram(consensus_matrix):
    upper_triangle = []
    n_samples = consensus_matrix.shape[0]
    for row in range(n_samples - 1):
        for element in list(consensus_matrix[row, (row + 1):]):
            upper_triangle.append(element)

    plt.hist(upper_triangle, bins = 10, range = (0, 1))
    plt.ylabel("Frequency")
    plt.show()


def seriation(Z,N,cur_index):
    '''
        input:
            - Z is a hierarchical tree (dendrogram)
            - N is the number of points given to the clustering process
            - cur_index is the position in the tree for the recursive traversal
        output:
            - order implied by the hierarchical tree Z
            
        seriation computes the order implied by a hierarchical tree (dendrogram)
    '''
    if cur_index < N:
        return [cur_index]
    else:
        left = int(Z[cur_index-N,0])
        right = int(Z[cur_index-N,1])
        return (seriation(Z,N,left) + seriation(Z,N,right))
    
def compute_serial_matrix(dist_mat, method="ward"):
    '''
        input:
            - dist_mat is a distance matrix
            - method = ["ward","single","average","complete"]
        output:
            - seriated_dist is the input dist_mat,
              but with re-ordered rows and columns
              according to the seriation, i.e. the
              order implied by the hierarchical tree
            - res_order is the order implied by
              the hierarhical tree
            - res_linkage is the hierarhical tree (dendrogram)
        
        compute_serial_matrix transforms a distance matrix into 
        a sorted distance matrix according to the order implied 
        by the hierarchical tree (dendrogram)
    '''
    N = len(dist_mat)
    flat_dist_mat = squareform(dist_mat)
    res_linkage = linkage(flat_dist_mat, method=method, preserve_input=True)
    res_order = seriation(res_linkage, N, N + N-2)
    seriated_dist = np.zeros((N,N))
    a,b = np.triu_indices(N,k=1)
    seriated_dist[a,b] = dist_mat[ [res_order[i] for i in a], [res_order[j] for j in b]]
    seriated_dist[b,a] = seriated_dist[a,b]
    
    return seriated_dist, res_order, res_linkage
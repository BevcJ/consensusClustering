import time
import hdbscan
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from utils import  bootstrap, conn_matrix, indi_matrix

class ConsensusClustering:
    def __init__(self, clustering_algorithm, n_iter):
        """Initialize consensus clustering"""
        self.clustering_algorithm = clustering_algorithm
        self.n_iter = n_iter



    def cc_fit(self, X, y):
        """Perform consesus clustering from input dataset
        Parameters
        ----------
        X : array of shape (n_samples, n_features),
        y : class labels for samples

        """

        n_samples, n_features = X.shape
        rand_scores = []

        idx = np.array([range(n_samples)]).reshape(n_samples, 1)
        X = np.hstack([idx, X, y])

        connectivity_matrix = np.zeros((n_samples, n_samples))
        indicator_matrix = np.zeros((n_samples, n_samples))

        time_taken = 0
        start_time = time.time()
        for iteration in range(self.n_iter):

            resampled_ds, rows_list = bootstrap(X, int(0.9 * n_samples))
            distance_matrix = pairwise_distances(resampled_ds[:, 1:-1], metric='cosine')
            clustering = self.clustering_algorithm.fit(distance_matrix)

            #clustering = self.clustering_algorithm.fit(resampled_ds[:, 1:-1])
            predicted_labels = clustering.labels_

            # Make dictionary of Original indices of data for correct matrix operation
            idx_original = resampled_ds[:, 0].astype(int)
            idx_current = range(0, n_samples)
            dict_original2resampled_idx = dict(zip(idx_original, idx_current))

            # Assebmle matrices according to the article and sum them up at every iteration,
            # rather to store NUM_REPEATS in list and then sum up
            connectivity_matrix += conn_matrix(predicted_labels, rows_list,
                                               dict_original2resampled_idx,
                                               n_samples)
            indicator_matrix += indi_matrix(rows_list, n_samples)

            # Evalutaion
            true_labels = resampled_ds[:,-1]
            predicted_labels = clustering.labels_

            rand_score = adjusted_rand_score(true_labels, predicted_labels)
            rand_scores.append(rand_score)
            #print(rand_score)
            if iteration % 25 == 0:
                print("Done 25 iterations")

            if False:
                print("Starting iteration: {}".format(iteration))
                print("Adjusted rand score for this iteration: {}".format(rand_score))

        end_time = time.time()

        print("Time taken (average) for one iteration: {}".format((end_time - start_time) / self.n_iter))
        print("Total time taken in minutes: {}".format((end_time - start_time) / 60))
        print("Average adj. rand index: {}".format(np.average(rand_scores)))

        # Calculate conesnsus matirx based on equation (2) from article
        consensus_matrix = connectivity_matrix / indicator_matrix

        # Add also the upper diagonalto consensus_matrix
        consensus_matrix += consensus_matrix.T

        return consensus_matrix


if __name__ == "__main__":
    
    from sklearn import datasets
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import adjusted_rand_score
    from itertools import combinations
    from random import randint
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

   # DATA LOAD
    PANCERAS_HUMAN_SAMPLE = True
    PATH = r"C:\Users\Jakob\Documents\consensusClustering\experiment\datasets"

    if PANCERAS_HUMAN_SAMPLE:
        dataframe = pd.read_csv(PATH + '\pancreas_human_sample.tab', sep = '\t', low_memory = False)
        dataframe.drop([0, 1], inplace = True)

        # Save true labels and drop than unneede columns
        n_rows, n_columns = dataframe.shape
        y = dataframe['class'].values.reshape(n_rows, 1)

        dataframe.drop(['class', 'Selected'], axis = 1, inplace = True)
        X = dataframe.values 

    clustering_algorithm = DBSCAN(min_samples = 3, eps = 0.3, metric = 'cosine')
    consensus_clustering = ConsensusClustering(clustering_algorithm, 10)
    
    consensus_matrix = consensus_clustering.cc_fit(X, y)
    distance_matrix = 1 - consensus_matrix 

    clustering = DBSCAN(min_samples = 3, eps = 0.1, metric='precomputed').fit(distance_matrix)
    predicted_labels = clustering.labels_
    true_labels = y.flatten()
    print("Adjusted rand score for custering: {}"
                    .format(adjusted_rand_score(true_labels, predicted_labels)))
   
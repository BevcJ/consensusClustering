from sklearn.cluster import DBSCAN, KMeans
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
from process_data import load_panceras_mouse_3000, load_panceras_human_3000, load_mouse_retinal_3000
from utils import find_hyper_parameters, hdbscan_consensus_matrix


PATH = r"C:\Users\Jakob\Documents\consensusClustering\experiment\datasets_processed"
#PATH = r"C:\Users\Jakob\Documents\consensusClustering\experiment\datasets"
# FUNCTION_NAMES = [load_panceras_mouse]#, load_aml8k]
# DATASET_NAME = ["load_panceras_mouse"]#, "AML8K"]

FUNCTION_NAMES = [load_panceras_mouse_3000, load_panceras_human_3000, load_mouse_retinal_3000]#], load_panceras_human_3000]#, load_mouse_retinal_3000]
DATASET_NAME = ["load_panceras_mouse_3000", "load_panceras_human_3000", "load_mouse_retinal_3000"]#, "load_panceras_human_3000"]#", "load_mouse_retinal_3000"]

start = time.time()
for load_data, dataset_name in zip(FUNCTION_NAMES, DATASET_NAME):

    print("######## Loading Dataset: {} ###############".format(dataset_name))
    start1 = time.time()
    X, y = load_data(PATH)
    n_rows, n_columns = X.shape
    print("Data loaded successfuly, time taken: {}\n".format(time.time() - start1))
    print("Dataset size: {}".format(X.shape))
    print("---------------------------------------------------------------------\n")

    print("############# Normalizing dataset ###################")
    start1 = time.time()
    norm_data = normalize(X, norm='l2')
    print("Data normalized successfuly, time taken: {}\n".format(time.time() - start1))
    print("---------------------------------------------------------------------\n")
    # print("Calculating distance matrix and applying PCA")
    # start1 = time.time()

    # dist_matrix = pairwise_distances(norm_data, metric='cosine')
    # n_components = int(n_rows * 0.08) #0.05 is variable suggested in article 0.04 -0.07

    # sklearn_pca = sklearnPCA(n_components=n_components)
    # distance_matrix_PCA = sklearn_pca.fit_transform(dist_matrix)
    
    # norm_data = distance_matrix_PCA
    # print("Reduced distance matrix shape: {}".format(norm_data.shape))
    # print("Finished successfuly, time taken: {}\n".format(time.time() - start1))



    

    best_cluster_size = find_hyper_parameters(norm_data, y, 45, 1)

    # use best hyperparameters from single run test
    #clustering_algorithm = hdbscan.HDBSCAN(min_cluster_size=best_cluster_size)

    #clustering_algorithm = kmeans = KMeans(n_clusters=13).fit(X)

    clustering_algorithm = hdbscan.HDBSCAN(min_cluster_size=best_cluster_size, metric='precomputed')
    # clustering_algorithm = DBSCAN(min_samples = 3, eps = 0.3, metric = 'cosine')
    consensus_clustering = ConsensusClustering(clustering_algorithm, 250)
    consensus_matrix = consensus_clustering.cc_fit(norm_data, y)

    file_name = "matrices\\" + dataset_name + "_final_250_iter"
    np.save(file_name, consensus_matrix)

    #hdbscan_consensus_matrix(consensus_matrix, 70, 1, y)

    print("\n\n")

end = time.time()
print("Time taken: {}".format(end - start))

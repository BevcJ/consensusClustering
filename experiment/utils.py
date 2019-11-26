from itertools import combinations
from random import randint

import numpy as np

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
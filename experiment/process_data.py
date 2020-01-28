import pandas as pd
import numpy as np

from sklearn.preprocessing import normalize


def load_panceras_mouse(PATH):
    dataframe = pd.read_csv(PATH + '\pancreas_cells_mouse_2016.tab',
                            sep='\t', low_memory=False)
    dataframe.drop([0, 1], inplace=True)

    # Save true labels and drop than unneede columns
    n_rows, n_columns = dataframe.shape
    y = dataframe['assigned_cluster'].values.reshape(n_rows, 1)

    dataframe.drop(['class', 'assigned_cluster',
                    'barcode', 'level_0', 'source', 
                    'Selected'], axis=1, inplace=True)
          
    X = dataframe.values
    X = normalize(X, norm='l2')

    return X.astype(float), y


def load_panceras_mouse_3000(PATH):
    dataframe = pd.read_csv(PATH + '\Baron2016_panceras_cells_mouse_3000.tab',
                            sep='\t', low_memory=False)
    dataframe.drop([0, 1], inplace=True)

    # Save true labels and drop than unneede columns
    n_rows, n_columns = dataframe.shape
    y = dataframe['assigned_cluster'].values.reshape(n_rows, 1)

    dataframe.drop(['class', 'assigned_cluster',
                    'barcode', 'level_0', 'source', 
                    'Selected'], axis=1, inplace=True)
               
    X = dataframe.values
    X = normalize(X, norm='l2')

    return X.astype(float), y


def load_aml8k(PATH):
    dataframe = pd.read_csv(PATH + r'\aml_8k.tab', sep = '\t', low_memory = False)
    dataframe.drop([0, 1], inplace = True)

    # Save true labels and drop than unneede columns
    n_rows, n_columns = dataframe.shape
    y = dataframe['Type'].values.reshape(n_rows, 1)

    dataframe.drop(['Type','Barcode', 'Replicate', 'ID', 'Selected'], axis = 1, inplace = True)
    X = dataframe.values

    return X, y


def load_panceras_human(PATH):
    dataframe = pd.read_csv(PATH + '\panceras_cells_human_2016.tab',
                            sep='\t', low_memory=False)
    dataframe.drop([0, 1], inplace=True)

    # Save true labels and drop than unneede columns
    n_rows, n_columns = dataframe.shape
    y = dataframe['class'].values.reshape(n_rows, 1)

    dataframe.drop(['class', 'Selected', 'barcode',
                    'Cell ID', 'Batch ID', 'Patient'], axis=1, inplace=True)
    X = dataframe.values
    print("Original dataset shape: {}, {}".format(n_rows, n_columns))

    return X, y


def load_panceras_human_3000(PATH):
    dataframe = pd.read_csv(PATH + '\Baron2016_panceras_cells_human_3000.tab',
                            sep='\t', low_memory=False)
    dataframe.drop([0, 1], inplace=True)

    # Save true labels and drop than unneede columns
    n_rows, n_columns = dataframe.shape
    y = dataframe['class'].values.reshape(n_rows, 1)

    dataframe.drop(['class', 'Selected', 'barcode',
                    'Cell ID', 'Batch ID', 'Patient'], axis=1, inplace=True)
    X = dataframe.values
    print("Original dataset shape: {}, {}".format(n_rows, n_columns))

    return X, y

def load_mouse_retinal_3000(PATH):
    dataframe = pd.read_csv(PATH + '\Shekar2016_mouse_retinal_bipolar_neurons_large_3000.tab',
                            sep='\t', low_memory=False)
    dataframe.drop([0, 1], inplace=True)

    # Save true labels and drop than unneede columns
    n_rows, n_columns = dataframe.shape
    y = dataframe['Cluster ID'].values.reshape(n_rows, 1)

    dataframe.drop(['Cluster ID', 'Selected', 'Cell ID'], axis = 1, inplace = True)
    X = dataframe.values
    print("Original dataset shape: {}, {}".format(n_rows, n_columns))

    return X, y


def gene_filter(X, percentage_rate=0.06):
    """Gene filter: reduces the feature space in dataset
       Based on Kiselev2017 paper.


    Keyword arguments:
    X -- numpy array of shape (n_samples, n_features)
    percentage_rate -- float betwen 0 and 1.0, default 0.06 
    """
    n_rows, n_columns = X.shape

    idx_list = []

    for idx in range(0, n_columns):
        column = X[:, idx]
       
        high_expression_share = np.sum(column >= 2) / n_rows
        low_expression_share = np.sum(column >= 0) / n_rows
    
        if (high_expression_share > percentage_rate) or (low_expression_share < (1 - percentage_rate)):
            idx_list.append(idx)

    return X[:, idx_list]


if __name__ == "__main__":
    matrix = [[1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [0, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 3],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 0],
                [1, 1, 0, 0, 1, 3]]

    # test_array = np.array(matrix)
    # print(test_array)
    # print(test_array.shape)

    # filtered_array = gene_filter(test_array)
    # print(filtered_array)
    # print(filtered_array.shape)


    PATH = r"C:\Users\Jakob\Documents\consensusClustering\experiment\datasets"
    X,y = load_panceras_mouse(PATH)
    print(X.shape)
    print(X[0:5,:])
    filtered_array = gene_filter(X)
    
    
    print(filtered_array.shape)
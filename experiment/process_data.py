import pandas as pd


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

    return X, y


def load_aml8k(PATH):
    dataframe = pd.read_csv(PATH + r'\aml_8k.tab', sep = '\t', low_memory = False)
    dataframe.drop([0, 1], inplace = True)

    # Save true labels and drop than unneede columns
    n_rows, n_columns = dataframe.shape
    y = dataframe['Type'].values.reshape(n_rows, 1)

    dataframe.drop(['Type','Barcode', 'Replicate', 'ID', 'Selected'], axis = 1, inplace = True)
    X = dataframe.values

    return X, y

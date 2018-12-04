# The PreprocessingHandler contains methods for cleaning and preprocessing the data prior to analyzing it.

from sklearn.preprocessing import StandardScaler


# give list of columns to drop
def drop_columns(dataset, column_list):
    for column in column_list:
        dataset = dataset.drop(column, axis=1)
    return dataset


# split dataset into data and target
def get_X_y(dataset):
    target = dataset.iloc[:, :1]
    X = dataset.iloc[:, 1:]
    return X, target


# standardize data
def standard(X):
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    return X_trans

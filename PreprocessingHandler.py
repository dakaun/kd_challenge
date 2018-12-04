import IOHandler
from ast import literal_eval
# The PreprocessingHandler contains methods for cleaning and preprocessing the data prior to analyzing it.

from sklearn.preprocessing import StandardScaler
import numpy as np


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

# transform genre dictionary into columnwise entries
# https://chrisalbon.com/machine_learning/vectors_matrices_and_arrays/converting_a_dictionary_into_a_matrix/
# https://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary
def transform_dictionary(df):
    genre_df = df['genres']
    for row in genre_df:
        row = literal_eval(row)
        for dic__pair in row:
            for dic_element in dic__pair:
                for key in dic_element:
                    if key == "name":
                        dic__pair.pop(key, None)
    print(row)

    # Create DictVectorizer object
    #dictvectorizer = DictVectorizer(sparse=False)

    # Convert dictionary into feature matrix
    # features = dictvectorizer.fit_transform(genre_df)

    # View feature matrix
    #features

data = IOHandler.read_data()
transform_dictionary(data)

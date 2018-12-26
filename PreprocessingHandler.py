# The PreprocessingHandler contains methods for cleaning and preprocessing the data prior to analyzing it.
import ast
import json
import pandas as pd
import IOHandler
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
import numpy as np


# extract numerical columns
def extract_numerical(dataset):
    subset = dataset[['label', 'budget', 'popularity', 'revenue', 'runtime']]
    return subset


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

def transform_dictionary(df_column, df_name):
    '''
    :param df: excepts column with dictionary entries like 'genres'
    :return: one hot encoding matrix with the name entries as column and rows stay the same
    '''
    row_index = 0
    result_df = pd.DataFrame()

    for row in df_column:
        row_json = json.loads(row)
        for element in row_json:
            result_df.loc[row_index, element['name']] = 1
        row_index += 1
    result_df.to_csv('./data/' + df_name + '.csv')
    return result_df



df = IOHandler.read_data()
genre_df = df['genre']
transform_dictionary(genre_df, 'genre')

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
    target = dataset['label']
    X = dataset.loc[:, dataset.columns != 'label']
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
    :param df_name: name for csv file
    :return: one hot encoding matrix with the name entries as column and rows stay the same
    '''
    row_index = 0
    result_df = pd.DataFrame()

    for row in df_column:
        row_json = json.loads(row)
        for element in row_json:
            result_df.loc[row_index, element['name']] = 1
        row_index += 1
    result_df.fillna(0)
    result_df.to_csv('./data/' + df_name + '.csv', index=False)
    return result_df


def write_df(df, name):
    df.to_csv('./data/' + name + '.csv', index=False)

# todo adapt to more than two df
def combine_df(df1, df2):
    combined_df = pd.concat([df1, df2], axis=1)
    return combined_df

data = IOHandler.read_data('train')

# # -- run to get genre csv as one hot encod (21 columns)
# genre_df = data['genres']
# trans_genre_df = transform_dictionary(genre_df, 'genres')

# # -- run to get csv with only numerical columns
# num_data = extract_numerical(data)
# write_df(num_data, 'numerical_df')

# # -- run to get one csv with production companies as one hot encod - but more than 2500 columns
# prod_comp = data['production_companies']
# transf_prod_comp = transform_dictionary(prod_comp, 'prod_companies')

# # -- run to get one csv with production countries as one hot encod (58 columns)
# prod_countries = data['production_countries']
# transf_prod_countries = transform_dictionary(prod_countries, 'prod_countries')

# # # -- run to get one csv with languages as one hot encod (53 columns)
# lang = data['spoken_languages']
# transf_lang = transform_dictionary(lang, 'spoken_languages')

# num_genre_df = combine_df(trans_genre_df, num_data)
# write_df(num_genre_df, 'num_genre')
# The PreprocessingHandler contains methods for cleaning and preprocessing the data prior to analyzing it.
import ast
import json
import pandas as pd
import IOHandler as io
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import EvaluationHandler as ev

# todo date transformieren + hinzufügen
# todo pca
# todo leere felder

# extract numerical columns
def extract_numerical(dataset):
    subset = dataset[['label', 'budget', 'popularity', 'revenue', 'runtime']]
    return subset


# give list of columns to drop
def drop_columns(dataset, column_list):
    '''
    :param dataset:
    :param column_list: list of column which needs to be deleted
    :return:
    '''
    for column in column_list:
        dataset = dataset.drop(column, axis=1)
    return dataset


# split dataset into data and target
def get_X_y(dataset):
    '''
    splits dataset into target (which is the label category) and dataset (which is every column except the label column)
    :param dataset:
    :return:
    '''
    target = dataset['label']
    X = dataset.loc[:, dataset.columns != 'label']
    return X, target


# standardize data
def standard(X):
    scaler = StandardScaler()
    X_trans = scaler.fit_transform(X)
    return X_trans


# transform genre dictionary into columnwise entries
def transform_dictionary(df_column): # todo leere dictionaries
    '''
    :param df: excepts column with dictionary entries like 'genres'
    :return: one hot encoding matrix with the name entries as column and rows stay the same
    '''
    row_index = 0
    result_df = pd.DataFrame()

    for row_index, row in enumerate(df_column):
        row_json = json.loads(row)
        if row_json:
            for element in row_json:
                result_df.loc[row_index, element['name']] = 1
        else:
            result_df.loc[row_index] = 0
    result_df.fillna(value=0, inplace=True)
    print(result_df.shape)
    # result_df.to_csv('./data/' + df_name + '.csv', index=False) - done seperately through IO Method
    return result_df


def combine_df(dataframe_list):
    '''
    :param dataframe_list: get dataframes in a list like [df1, df2]
    :return: one dataframe which combines all given df
    '''
    combined_df = pd.concat(dataframe_list, axis=1)
    return combined_df


def hot_encode_and_write_all_columns(df, name):
    '''
    :param df: dataframe which columns should be hot encoded
    :param name: name of the data that should be transformed, should be either "train" or "test"
                ALSO: suffix of csv for better identification
    :return: one dataframe which combines all given df
    '''

    # -- run to get genre csv as one hot encod (21 columns)
    genre_df = df['genres']
    trans_genre_df = transform_dictionary(genre_df)
    io.write_df(trans_genre_df, 'genres' + "_" + name)

    # -- run to get csv with only numerical columns
    num_data = extract_numerical(df)
    io.write_df(num_data, 'numerical_df' + "_" + name)

    # -- run to get one csv with production companies as one hot encod - but more than 2500 columns
    prod_comp = df['production_companies']
    transf_prod_comp = transform_dictionary(prod_comp)
    io.write_df(transf_prod_comp, 'prod_companies' + "_" + name)

    # -- run to get one csv with production countries as one hot encod (58 columns)
    prod_countries = df['production_countries']
    transf_prod_countries = transform_dictionary(prod_countries)
    io.write_df(transf_prod_countries, 'prod_countries' + "_" + name)

    # -- run to get one csv with languages as one hot encod (53 columns)
    lang = df['spoken_languages']
    transf_lang = transform_dictionary(lang)
    io.write_df(transf_lang, 'spoken_languages' + "_" + name)

    keywords = df['keywords']
    transf_keywords = transform_dictionary(keywords)
    io.write_df(transf_keywords, 'keywords_' + name)

    # num_genre_df = combine_df([trans_genre_df, num_data])
    # io.write_df(num_genre_df, 'num_genre' + "_" + name)


def dim_reduction_components(df, n_components, visualize=False):
    '''
    :param df: dataframe that which dimensionality should be reduced
    :param visualize: if true plots of the explained variance of the principal components are plotted
    :return: reduced matrix, depending on the chosen number of principal components
    '''

    pca = PCA(n_components=n_components)
    reduced_matrix = pca.fit_transform(df)

    if visualize == True:
        ev.pca_var_exp_visualizer(pca)

    # Todo: add functionality to extract pca model in order to transform test data as well

    return reduced_matrix


def dim_reduction_var_exp(df, n_components, visualize=False):
    '''
    :param df: dataframe that which dimensionality should be reduced
    :param visualize: if true plots of the explained variance of the principal components are plotted
    :return: reduced matrix, depending on the chosen number of principal components
    '''

    pca = PCA(n_components=n_components, svd_solver='full')
    reduced_matrix = pca.fit_transform(df)

    if visualize == True:
        ev.pca_var_exp_visualizer(pca)

    #Todo: add functionality to extract pca model in order to transform test data as well

    return reduced_matrix
# The IO Handler (InputOutput handler) contains methods import various dataset (e.g. train data set, additional
# feature dataset) to be later used within the ML file

import csv
import os.path
import pickle
import numpy as np
import pandas as pd

base_path = './data/'
results_filename = 'test_predict'
filetype = ".csv"


def read_data(data_name):
    df = pd.read_csv('./data/' + data_name + '.csv')
    return df


def write_df(df, name):
    '''
    :param df: df which needs to be written to a file
    :param name: name for the file
    :return: csv file of the df with the given name in the data folder
    '''
    df.to_csv('./data/' + name + '.csv', index=False)


def append_results_to_csv(df):
    path_to_file = base_path + results_filename + filetype

    result_csv_content = pd.read_csv('./data/' + 'test' + '.csv')
    df = pd.concat([result_csv_content, df], axis=1)
    df.to_csv(path_to_file)
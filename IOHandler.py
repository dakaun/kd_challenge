# The IO Handler (InputOutput handler) contains methods import various dataset (e.g. train data set, additional
# feature dataset) to be later used within the ML file

import csv
import os.path
import pickle
import numpy as np
import pandas as pd


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
# The IO Handler (InputOutput handler) contains methods import various dataset (e.g. train data set, additional
# feature dataset) to be later used within the ML file

import csv
import os.path
import pickle
import numpy as np
import pandas as pd


def read_data():
    df = pd.read_csv('./data/train.csv')
    return df

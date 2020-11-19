"""
module to start exploring fall data
"""
import sys
import pdb
import time
import datetime as dt
import pandas as pd


def load_data(path="../data/fall_har_up.csv"):
    """
    load the data from the csv
    """
    fall_df = pd.read_csv(path)
    pdb.set_trace()
    return fall_df
    

if __name__ == '__main__':
    load_data()
    
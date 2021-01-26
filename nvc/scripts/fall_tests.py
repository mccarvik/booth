"""
module to start exploring fall data
"""
import sys
import pdb
import time
import numpy as np
import datetime as dt
import pandas as pd

from sklearn.linear_model import LogisticRegression

from feature_selection import sbs_run


def load_data(path="../data/fall_har_up.csv", csv=False):
    """
    load the data from the csv
    """
    fall_df = pd.read_csv(path)
    # drop non wrist columns
    column_list = [n for n in range(1,29)] + [n for n in range(36,43)]
    fall_df = fall_df.drop(columns=fall_df.columns[column_list])
    fall_df = fall_df.drop(index=[0])
    fall_df.columns = ['time', 'wristaccel_x', 'wristaccel_y', 'wristaccel_z', 'wristangle_x', 
                       'wristangle_y', 'wristangle_z', 'wristlumin', 'subject', 'activity',
                       'trial', 'tag']
    if csv:
        fall_df.to_csv("../data/fall_df.csv")
    return fall_df


def run_ml():
    data = load_data()
    inputs = ['wristaccel_x', 'wristaccel_y', 'wristaccel_z', 'wristangle_x', 'wristangle_y', 'wristangle_z', 'wristlumin']
    feature_selection(data, inputs)
    

def feature_selection(data, inputs):
    pdb.set_trace()
    sbs_run(data, tuple(inputs), est=LogisticRegression(C=100, random_state=0, penalty='l1'))



    

if __name__ == '__main__':
    run_ml()    
    
    
    
    
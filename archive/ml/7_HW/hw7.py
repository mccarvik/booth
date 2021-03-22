import pdb
import csv
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from scikitplot.helpers import cumulative_gain_curve
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
import scipy.stats as scipy
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier

def get_data():
    data = []
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    return train, test


def neural_net(data):
    # To train
    # layers
    # activation
    # alpha
    
    
    clf = MLPClassifier(solver='sgd', alpha=1e-5, tol=1e-4, hidden_layer_sizes=(6, 6), max_iter=200, 
                        learning_rate_init=1e-2, random_state=1, early_stopping=False)
    train = data[0]
    test = data[1][:-1]
    train['Activity'], act_ind = pd.factorize(train['Activity'])
    test['Activity'], act_ind = pd.factorize(test['Activity'])
    
    x = train.drop(columns=['Activity'])
    y = train['Activity']
    xt = test.drop(columns=['Activity'])
    yt = test['Activity']
    clf.fit(x, y)
    # pdb.set_trace()
    print(clf.predict(x))
    print("Train Accuracy")
    print(accuracy_score(clf.predict(x), y))
    print("Test Accuracy")
    print(accuracy_score(clf.predict(xt), yt))



def models(data):
    train = data[0]
    test = data[1][:-1]
    train['Activity'], act_ind = pd.factorize(train['Activity'])
    test['Activity'], act_ind = pd.factorize(test['Activity'])
    
    x = train.drop(columns=['Activity'])
    y = train['Activity']
    xt = test.drop(columns=['Activity'])
    yt = test['Activity']
    
    boost = GradientBoostingClassifier(n_estimators=20, random_state=1, max_leaf_nodes=200, 
                                      max_depth=20, min_samples_split=5, max_features=100)
    # boost = GradientBoostingClassifier(n_estimators=5)
    boost.fit(x, y)
    boost_pred = boost.predict(x)
    print("Boost Model:")
    print("Train Accuracy")
    print(accuracy_score(boost.predict(x), y))
    print("Test Accuracy")
    print(accuracy_score(boost.predict(xt), yt))
    
    rfc = RandomForestClassifier(n_estimators=50, random_state=1, max_leaf_nodes=200, 
                                 max_depth=30, n_jobs=1)
    rfc = RandomForestClassifier(n_estimators=5) 
    rfc.fit(x, y)
    print("RFC Model:")
    print("Train Accuracy")
    print(accuracy_score(rfc.predict(x), y))
    print("Test Accuracy")
    print(accuracy_score(rfc.predict(xt), yt))
    
    svc = SVC(C=10, kernel='rbf', tol=1e-3)
    svc.fit(x, y)
    print("SVC Model:")
    print("Train Accuracy")
    print(accuracy_score(svc.predict(x), y))
    print("Test Accuracy")
    print(accuracy_score(svc.predict(xt), yt))



if __name__ == '__main__':
    data = get_data()
    # neural_net(data)
    models(data)
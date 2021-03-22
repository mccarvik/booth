import pdb
import csv
import random
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
import scipy.stats as scipy
from sklearn.svm import SVC
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier


def get_data():
    train = pd.read_csv("paper_data_train_small.csv")
    test = pd.read_csv("paper_data_test_small.csv")
    return train, test


def booster(train, test):
    """
    """
    x = train.drop(columns=['fall_activity'])
    y = train['fall_activity']
    xt = test.drop(columns=['fall_activity'])
    yt = test['fall_activity']
    
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(x)
    x_train = scaling.transform(x)
    x_test = scaling.transform(xt)
    random.seed(42)
    index = random.sample(range(0, len(x_train)), 1500)
    x_train = x_train[index]
    y_train = y[index]
    
    # boost = OneVsRestClassifier(BaggingClassifier(SVC(kernel='linear'), max_samples=1.0 / 10, n_estimators=10))    
    boost = SVC(kernel='linear', class_weight={0:1, 1:4.2})   
    # boost = SVC(kernel='linear')
    boost.fit(x_train, y_train)
    boost_pred = boost.predict(x_train)
    print("Support Vector Machine:")
    print("Train Accuracy:   ", end="")
    print(accuracy_score(boost.predict(x_train), y_train))
    print("Test Accuracy:   ", end="")
    print(accuracy_score(boost.predict(x_test), yt))
    
    boost = GradientBoostingClassifier(n_estimators=1)
    # boost = GradientBoostingClassifier(n_estimators=5)
    boost.fit(x, y)
    boost_pred = boost.predict(x)
    print("Gradient Boosting Model:")
    print("Train Accuracy:   ", end="")
    print(accuracy_score(boost.predict(x), y))
    print("Test Accuracy:   ", end="")
    print(accuracy_score(boost.predict(xt), yt))
    
    boost = RandomForestClassifier(n_estimators=10)
    boost.fit(x, y)
    boost_pred = boost.predict(x)
    print("Random Forest:")
    print("Train Accuracy:   ", end="")
    print(accuracy_score(boost.predict(x), y))
    print("Test Accuracy:   ", end="")
    print(accuracy_score(boost.predict(xt), yt))
    
    boost = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    boost.fit(x, y)
    boost_pred = boost.predict(x)
    print("KNearest Neighbor:")
    print("Train Accuracy:   ", end="")
    print(accuracy_score(boost.predict(x), y))
    print("Test Accuracy:   ", end="")
    print(accuracy_score(boost.predict(xt), yt))
    
    boost = LogisticRegression(class_weight={0:1, 1:3.5})
    boost.fit(x, y)
    boost_pred = boost.predict(x)
    print("Logistic Regression:")
    print("Train Accuracy:   ", end="")
    print(accuracy_score(boost.predict(x), y))
    print("Test Accuracy:   ", end="")
    print(accuracy_score(boost.predict(xt), yt))
    pdb.set_trace()
    boost_probs = boost.predict_proba(xt).T[1]
    fpr,tpr,_ = roc_curve(yt, boost_probs)
    plt.plot(fpr,tpr,c='r', label='LogRegr')
    plt.plot([0,1], [0,1], c='b')
    plt.xlabel("False Pos")
    plt.xlabel("True Pos")
    plt.legend()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()
    
    
    
    # boost = MLPClassifier(solver='sgd', alpha=1e-5, tol=1e-4, hidden_layer_sizes=(6, 6), max_iter=200, 
    #                     learning_rate_init=1e-2, random_state=1, early_stopping=False)
    # boost.fit(x, y)
    # boost_pred = boost.predict(x)
    # print("Neural Network:")
    # print("Train Accuracy:   ", end="")
    # print(accuracy_score(boost.predict(x), y))
    # print("Test Accuracy:   ", end="")
    # print(accuracy_score(boost.predict(xt), yt))
    
    

if __name__ == '__main__':
    train, test = get_data()
    booster(train, test)
    
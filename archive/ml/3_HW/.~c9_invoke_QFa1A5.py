"""
HW3 ML
"""

import pdb
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.inspection import plot_partial_dependence
from sklearn.ensemble import RandomForestRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
import scipy.stats as scipy



def q2(train, test):
    for cat in ['humidity', 'temp', 'atemp', 'windspeed']:
        plt.scatter(train[cat], train['count'], s=5, facecolors='none', edgecolors='r')
        plt.xlabel(cat)
        plt.ylabel("count")
        plt.savefig('png/scatter_' + cat + '.png', dpi=300)
        plt.close()
    
    train[['count','season']].boxplot(by="season")
    plt.savefig('png/seasonbox.png', dpi=300)
    plt.close()

    work = train[train['workingday'] == 1]
    plt.scatter(work['hour'], work['count'], s=5, facecolors='none', edgecolors='r', label="Working Day")
    plt.xlabel("hour")
    plt.ylabel("count")
    plt.legend()
    plt.savefig('png/workdayscatter.png', dpi=300)
    plt.close()
    
    work = train[train['workingday'] == 0]
    plt.scatter(work['hour'], work['count'], s=5, facecolors='none', edgecolors='b', label="Not Working Day")
    plt.xlabel("hour")
    plt.ylabel("count")
    plt.legend()
    plt.savefig('png/notworkdayscatter.png', dpi=300)
    plt.close()
    
    work = train[train['atemp'] > 23]
    plt.scatter(work['hour'], work['count'], s=5, facecolors='none', edgecolors='r', label="Atemp > 23")
    work = train[train['atemp'] < 23]
    plt.scatter(work['hour'], work['count'], s=5, facecolors='none', marker="^", edgecolors='b', label="Atemp < 23")
    plt.xlabel("hour")
    plt.ylabel("count")
    plt.legend()
    plt.savefig('png/atemphoursscatter.png', dpi=300)
    plt.close()
    
    for season in [4, 1, 2, 3]:
        work = train[train['season'] == season]
        label = "season = " + str(season)
        plt.scatter(work['hour'], work['count'], s=5, facecolors='none', edgecolors='r', label=label)
        plt.xlabel("hour")
        plt.ylabel("count")
        plt.legend()
        plt.savefig('png/seasonscatter' + str(season) + '.png', dpi=300)
        plt.close()
    
    
    work = train[train['year'] == 2011]
    plt.scatter(work['hour'], work['count'], s=5, facecolors='none', edgecolors='b', label="2011")
    plt.xlabel("hour")
    plt.ylabel("count")
    plt.legend()
    plt.savefig('png/yearscatter2011.png', dpi=300)
    plt.close()
    
    work = train[train['year'] == 2012]
    plt.scatter(work['hour'], work['count'], s=5, facecolors='none', edgecolors='b', label="2012")
    plt.xlabel("hour")
    plt.ylabel("count")
    plt.legend()
    plt.savefig('png/yearscatter2012.png', dpi=300)
    plt.close()


def q22(train, test):
    rfc = RandomForestRegressor(n_estimators=200, random_state=1, n_jobs=1)
    features = ['windspeed', 'humidity', 'temp', 'atemp']
    work = train[features]
    rfc.fit(work, train['count'])
    # plot_partial_dependence(rfc, work, features)
    # plt.savefig('png/partial_dep.png', dpi=500)
    # plt.close()
    feats = rfc.feature_importances_
    x_pos = [i for i, _ in enumerate(feats)]
    plt.bar(x_pos, feats)
    plt.xticks(x_pos, features)
    plt.savefig('png/feat_rfc.png', dpi=500)
    plt.close()
    
    rfc = GradientBoostingRegressor(n_estimators=200, random_state=1)
    features = ['windspeed', 'humidity', 'atemp']
    # features = ['windspeed', 'humidity', 'temp', 'atemp']
    work = train[features]
    rfc.fit(work, train['count'])
    # plot_partial_dependence(rfc, work, features, ax=ax)
    # plt.savefig('png/partial_dep_boost.png', dpi=500)
    # plt.close()
    feats = rfc.feature_importances_
    x_pos = [i for i, _ in enumerate(feats)]
    plt.bar(x_pos, feats)
    plt.xticks(x_pos, features)
    # plt.savefig('png/feat_rfc_boost.png', dpi=500)
    plt.savefig('png/feat_rfc_boost_atemp.png', dpi=500)
    plt.close()

    
   
    # for feat in features:
    #     work = train[feat]
    #     dc = DecisionTreeRegressor()
    #     dc.fit(work.values.reshape(-1, 1), train['count'])
    #     x_vals = np.linspace(0,max(train[feat]), 100)
    #     y = dc.predict(x_vals.reshape(-1, 1))
    #     plt.plot(x_vals, y, "--", c="b", label=feat)
    #     # plt.scatter(work, train['count'], s=5, facecolors='none', edgecolors='b', label=feat)
    #     plt.legend()
    #     plt.savefig('png/scatter_best_fit' + feat + '.png', dpi=300)
    #     plt.close()


def q23(train, test):
    x_train, x_test, _, _ = train_test_split(train, train['count'], test_size=0.25, random_state=3)
    # features = list(x_train.drop(columns=['day']).columns)
    features = ['windspeed', 'humidity', 'temp', 'atemp']
    print("MSE")
    for feat in features:
        if feat == "count":
            continue
        work = x_train[feat]
        dc = DecisionTreeRegressor()
        dc.fit(work.values.reshape(-1, 1), x_train['count'])
        mse = mean_squared_error(x_test['count'], dc.predict(x_test[feat].values.reshape(-1,1)))
        print("{}:  {}".format(feat, str(mse)))


def get_data():
    train = pd.read_csv("./Bike_train.csv")
    test = pd.read_csv("./Bike_test.csv")
    return train, test
    

def dataclean(train, test):
    
    # Remove odd data
    train_t = train[['atemp', 'temp']]
    train_t['diff'] = train['atemp'] - train['temp']
    train_t = train_t[['diff']]
    zs = scipy.zscore(train_t)
    abs_z_scores = np.abs(zs)
    filtered_entries = (abs_z_scores < 4).all(axis=1)
    train = train[filtered_entries]
    
    pdb.set_trace()
    train_t = train[['humidity', 'temp']]
    train_t['diff'] = train['humidity'] - train['temp']
    train_t = train_t[['diff']]
    zs = scipy.zscore(train_t)
    abs_z_scores = np.abs(zs)
    filtered_entries = (abs_z_scores < 4).all(axis=1)
    train = train[filtered_entries]
    
    pdb.set_trace()
    ret = []
    train = train[train['weather'] != 4]
    # train['mrushh'] = train.apply(lambda x: 1 if x['hour'] in [7, 8] else 0, axis=1)
    # train['nrushh'] = train.apply(lambda x: 1 if x['hour'] in [16, 17, 18, 19] else 0, axis=1)
    # train['dayh'] = train.apply(lambda x: 1 if x['hour'] in [9, 10, 11, 12, 13, 14, 15] else 0, axis=1)
    # train['nighth'] = train.apply(lambda x: 1 if x['hour'] in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6] else 0, axis=1)
    for i in range(0, 24):
        train['h'+str(i)] = train.apply(lambda x: 1 if x['hour'] == i else 0, axis=1)

    train['w1'] = train.apply(lambda x: 1 if x['weather'] == 1 else 0, axis=1)
    train['w2'] = train.apply(lambda x: 1 if x['weather'] == 2 else 0, axis=1)
    train['w3'] = train.apply(lambda x: 1 if x['weather'] == 3 else 0, axis=1)
    
    for i in range(0, 24):
        test['h'+str(i)] = test.apply(lambda x: 1 if x['hour'] == i else 0, axis=1)

    test['w1'] = test.apply(lambda x: 1 if x['weather'] == 1 else 0, axis=1)
    test['w2'] = test.apply(lambda x: 1 if x['weather'] == 2 else 0, axis=1)
    test['w3'] = test.apply(lambda x: 1 if x['weather'] == 3 else 0, axis=1)
    
    return train, test


def model(train, test):
    train, test = dataclean(train, test)
    # get cross val score
    # work = train.drop(columns=["day", "count", "year", "season", "temp"])
    # for nest in range(840, 1200, 10):
    #     rfc = GradientBoostingRegressor(n_estimators=nest, random_state=1)
    #     scores = cross_val_score(rfc, work, train['count'], cv=5, scoring='neg_mean_squared_error')
    #     print("{}:  {}".format(nest, str(scores.mean())))
    
    feats = ['daylabel', 'workingday', 'atemp', 'humidity', 'w3', 'windspeed', 'hour']
    feats = feats + ["h" + str(i) for i in range(0,24)]
    for cat in ['h1', 'h5', 'h2', 'h3', 'h21', 'h22', 'h4', 'h0', 'h20', 'h14', 'h15', 'h12']:
        feats.remove(cat)
        
   
    # rfc = KNeighborsRegressor(n_neighbors=500, p=2, metric='minkowski')
    # rfc = RandomForestRegressor(n_estimators=1500, random_state=1, n_jobs=1)
    # x_train, x_test, _, _ = train_test_split(train, train['count'], test_size=0.25, random_state=3)
    x_train, x_test, _, _ = train_test_split(train, train['count'], test_size=0.25)
    work = train[feats]
    
    # pdb.set_trace()
    # max_features
    # for est in range(10, 100, 10):
    #     rfc = GradientBoostingRegressor(n_estimators=1500, max_depth=6, random_state=1,
    #                                     max_leaf_nodes=70, min_samples_split=est)
    #     scores = cross_val_score(rfc, work, train['count'], cv=5, scoring='neg_mean_squared_error')
    #     print("{}:  {}".format(est, str(scores.mean())))
    pdb.set_trace()
    work = x_train[feats]
    rfc = GradientBoostingRegressor(n_estimators=900, max_depth=5, max_leaf_nodes=70, random_state=1)
    rfc.fit(work, x_train['count'])
    mse = mean_squared_error(x_test['count'], rfc.predict(x_test[work.columns]))
    print("MSE:  {}".format(mse))
    feats = rfc.feature_importances_
    feats = dict(sorted(zip(work.columns, feats), key=lambda item: -item[1]))
    for k, v in feats.items():
        print(k + ":  " + str(v))
    
    work = train[feats]
    test = test[work.columns]
    rfc.fit(work, train['count'])
    create_csv(rfc, test)

def create_csv(model, test):
    y_pred = pd.DataFrame(model.predict(test)).reset_index()
    y_pred.columns = ["Id", "count"]
    y_pred['Id'] = y_pred['Id'] + 1
    y_pred = y_pred.set_index("Id")
    pdb.set_trace()
    y_pred['count'] = y_pred.apply(lambda x: 10 if x['count'] < 0 else x['count'], axis=1)
    y_pred.to_csv("sampleSubmission.csv")


if __name__ == '__main__':
    train, test = get_data()
    # q2(train, test)
    # q22(train, test)
    # q23(train, test)
    model(train, test)
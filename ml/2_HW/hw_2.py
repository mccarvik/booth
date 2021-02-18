"""
HW2 ML
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

import warnings
warnings.filterwarnings("ignore")


def q1(data):
    """
    """
    y = data['price']
    x = data.drop(columns=["price"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)
    
    train = x_train
    train['price'] = y_train
    
    # part 1.3
    pdb.set_trace()
    reg = sm.ols(formula="price ~ mileage", data=train).fit()
    beta = reg.params[1]
    alpha = reg.params['Intercept']
    x_vals = np.linspace(0,165000,100)
    plt.scatter(train['mileage'], train['price'], s=5, facecolors='none', edgecolors='r')
    line = x_vals * beta + alpha
    plt.plot(x_vals, line, "--", c="b")
    plt.savefig('scatter' + '.png', dpi=300)
    plt.close()

    # part 1.4
    mse_scores = []
    for degree in range(1,7):
        polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
        # polyreg.fit(train['mileage'].values.reshape(1,-1), train['price'].values)
        x = train['mileage'].values.reshape(-1,1)
        y = train['price'].values
        scores = cross_val_score(polyreg, x, y, cv=5, scoring='neg_mean_squared_error')
        # print("Degree: {}   scores: ".format(degree), end = "")
        print("Degree: {}: ".format(degree), end = "")
        # print(scores, end="")
        print("    mse: " + str(-1 * scores.mean()))
        mse_scores.append(scores.mean())
    plt.scatter(list(range(1,7)), [-1 * x for x in mse_scores], s=50, facecolors='none', edgecolors='r')
    plt.xlabel("degrees")
    plt.ylabel("MSE")
    plt.savefig('mse_degrees' + '.png', dpi=300)
    plt.close()
    
    # pdb.set_trace()
    # polynomial_features= PolynomialFeatures(degree=3)
    # xp = polynomial_features.fit_transform(train['mileage'].values.reshape(1,-1))
    # model = sm.OLS(train['price'], xp).fit()
    # ypred = model.predict(xp)
    # plt.scatter(x,y)
    # plt.plot(x,ypred)
    
    weights = np.polyfit(train['mileage'], train['price'], 3)
    # reg = sm.ols(formula="price ~ mileage + np.power(mileage, 2) + np.power(mileage, 3)", data=train).fit()
    # reg = sm.ols(formula="price ~ mileage + np.power(mileage, 2)", data=train).fit()

    # beta = reg.params[1]
    # beta2 = reg.params[2]
    # beta3 = reg.params[3]
    # alpha = reg.params['Intercept']
    beta = weights[2]
    beta2 = weights[1]
    beta3 = weights[0]
    alpha = weights[3]

    pdb.set_trace()
    x_vals = np.linspace(0,500000,100)
    plt.scatter(train['mileage'], train['price'], s=5, facecolors='none', edgecolors='r')
    line = x_vals * beta + x_vals**2 * beta2 + x_vals**3 * beta3 + alpha
    # line = x_vals * beta + x_vals**2 * beta2 + alpha
    plt.plot(x_vals, line, "--", c="b")
    plt.ylim(top=85000)
    plt.savefig('scatter_3d' + '.png', dpi=300)
    plt.close()
    
    return data


def q15(data):
    """
    """
    y = data['price']
    x = data.drop(columns=["price"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)
    
    train = x_train
    train['price'] = y_train
    
    # 1.5 KNN
    # log_k = []
    # mse_scores = []
    # x = train['mileage'].values.reshape(-1,1)
    # y = train['price'].values
    # for k in range(2,600):
    #     knn = KNeighborsRegressor(n_neighbors=k, p=2, metric='minkowski')
    #     scores = cross_val_score(knn, x, y, cv=5, scoring='neg_mean_squared_error')
    #     # print("Degree: {}   scores: ".format(degree), end = "")
    #     print("k: {}: ".format(k), end = "")
    #     # print(scores, end="")
    #     print("    mse: " + str(-1 * scores.mean()))
    #     mse_scores.append(-1 * scores.mean())
    
    # min_val = 100000000000
    # for k, mse in zip(range(2,600), mse_scores):
    #     if mse < min_val:
    #         min_val = mse
    #         num = k
    #     print("k = {}, mse = {}".format(str(k), str(mse)))
    # print("Best Number for k:  {},  MSE:  {}".format(num, min_val))
    
    # plt.plot(list(range(2,600)), mse_scores)
    # plt.xlabel("k")
    # plt.ylabel("mse")
    # plt.savefig('knn_mse' + '.png', dpi=300)
    # plt.close()
        
    # 1.5 Trees
    # mse_scores = []
    # x = train['mileage'].values.reshape(-1,1)
    # y = train['price'].values
    # for leaf in range(2,50):
    #     tree = DecisionTreeRegressor(max_leaf_nodes=leaf)
    #     scores = cross_val_score(tree, x, y, cv=5, scoring='neg_mean_squared_error')
    #     # print("Degree: {}   scores: ".format(degree), end = "")
    #     print("leaves: {}: ".format(leaf), end = "")
    #     # print(scores, end="")
    #     print("    mse: " + str(-1 * scores.mean()))
    #     mse_scores.append(-1 * scores.mean())
    
    # min_val = 100000000000
    # for leaf, mse in zip(range(2,50), mse_scores):
    #     if mse < min_val:
    #         min_val = mse
    #         num = leaf
    #     print("leaf = {}, mse = {}".format(str(leaf), str(mse)))
    # print("Best Number of Leaf Nodes:  {},  MSE:  {}".format(num, min_val))

    # plt.plot(list(range(2,50)), mse_scores)
    # plt.xlabel("max leaf nodes")
    # plt.ylabel("mse")
    # plt.savefig('tree_mse' + '.png', dpi=300)
    # plt.close()
    
     
    weights = np.polyfit(train['mileage'], train['price'], 3)
    beta = weights[2]
    beta2 = weights[1]
    beta3 = weights[0]
    alpha = weights[3]
    # reg = sm.ols(formula="price ~ mileage + np.power(mileage, 2)", data=train).fit()
    # beta = reg.params[1]
    # beta2 = reg.params[2]
    # alpha = reg.params['Intercept']
    x_vals = np.linspace(0,500000,100)
    plt.scatter(train['mileage'], train['price'], s=5, facecolors='none', edgecolors='r')
    line = x_vals * beta + x_vals**2 * beta2 + x_vals**3 * beta3 + alpha
    plt.plot(x_vals, line, "--", c="b", label="regression")
    
    knn = KNeighborsRegressor(n_neighbors=448, p=2, metric='minkowski')
    knn.fit(train['mileage'].values.reshape(-1, 1), y_train)
    y = knn.predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y, "--", c="k", label="knn")
    
    tree = DecisionTreeRegressor(max_leaf_nodes=15)
    tree.fit(train['mileage'].values.reshape(-1, 1), y_train)
    y = tree.predict(x_vals.reshape(-1, 1))
    plt.plot(x_vals, y, "--", c="lightgreen", label="dectrees")
    plt.ylim(top=85000)
    plt.legend()
    plt.savefig('scatter_best_fit' + '.png', dpi=300)
    plt.close()
    
    # knn mse
    knn_mse = mean_squared_error(y_test, knn.predict(x_test['mileage'].values.reshape(-1,1)))
    # dectree mse
    dectree_mse = mean_squared_error(y_test, tree.predict(x_test['mileage'].values.reshape(-1,1)))
    # reg mse
    pdb.set_trace()
    y_pred = x_test['mileage'] * beta + x_test['mileage']**2 * beta2 + x_test['mileage']**3 * beta3 + alpha
    reg_mse = mean_squared_error(y_test, y_pred)
    print("KNN MSE:  {}".format(str(knn_mse)))
    print("DEC TREE MSE:  {}".format(str(dectree_mse)))
    print("REG MSE:  {}".format(str(reg_mse)))
    

def q16(data):
    """
    """
    y = data['price']
    x = data.drop(columns=["price"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)
    
    train = x_train
    train['price'] = y_train
    
    # # 1.6 KNN
    # log_k = []
    # mse_scores = []
    # x = train['mileage'].values.reshape(-1,1)
    x = train[['mileage', 'year']]
    standardized_x = preprocessing.scale(x)
    xt = x_test[['mileage', 'year']]
    standardized_xt = preprocessing.scale(xt)
    y = train['price'].values
    # for k in range(2,200):
    #     knn = KNeighborsRegressor(n_neighbors=k, p=2, metric='minkowski')
    #     scores = cross_val_score(knn, standardized_x, y, cv=5, scoring='neg_mean_squared_error')
    #     # print("Degree: {}   scores: ".format(degree), end = "")
    #     print("k: {}: ".format(k), end = "")
    #     # print(scores, end="")
    #     print("    mse: " + str(-1 * scores.mean()))
    #     mse_scores.append(-1 * scores.mean())
    
    # min_val = 100000000000
    # for k, mse in zip(range(2,200), mse_scores):
    #     if mse < min_val:
    #         min_val = mse
    #         num = k
    #     print("k = {}, mse = {}".format(str(k), str(mse)))
    # print("Best Number for k:  {},  MSE:  {}".format(num, min_val))
    
    # plt.plot(list(range(2,200)), mse_scores)
    # plt.xlabel("k")
    # plt.ylabel("mse")
    # plt.savefig('knn_mse_year' + '.png', dpi=300)
    # plt.close()
        
    # # 1.5 Trees
    # mse_scores = []
    # x = train[['mileage', 'year']]
    # standardized_X = preprocessing.scale(x)
    # y = train['price'].values
    # for leaf in range(2,200):
    #     tree = DecisionTreeRegressor(max_leaf_nodes=leaf)
    #     scores = cross_val_score(tree, x, y, cv=5, scoring='neg_mean_squared_error')
    #     # print("Degree: {}   scores: ".format(degree), end = "")
    #     print("leaves: {}: ".format(leaf), end = "")
    #     # print(scores, end="")
    #     print("    mse: " + str(-1 * scores.mean()))
    #     mse_scores.append(-1 * scores.mean())
    
    # min_val = 100000000000
    # for leaf, mse in zip(range(2,200), mse_scores):
    #     if mse < min_val:
    #         min_val = mse
    #         num = leaf
    #     print("leaf = {}, mse = {}".format(str(leaf), str(mse)))
    # print("Best Number of Leaf Nodes:  {},  MSE:  {}".format(num, min_val))
    

    # plt.plot(list(range(2,200)), mse_scores)
    # plt.xlabel("max leaf nodes")
    # plt.ylabel("mse")
    # plt.savefig('tree_mse_year' + '.png', dpi=300)
    # plt.close()
    knn = KNeighborsRegressor(n_neighbors=59, p=2, metric='minkowski')
    knn.fit(standardized_x, y_train)
    
    tree = DecisionTreeRegressor(max_leaf_nodes=70)
    tree.fit(standardized_x, y_train)
    
    # knn mse
    pdb.set_trace()
    knn_mse = mean_squared_error(y_test, knn.predict(standardized_xt))
    # dectree mse
    dectree_mse = mean_squared_error(y_test, tree.predict(standardized_xt))
    # reg mse
    print("KNN MSE:  {}".format(str(knn_mse)))
    print("DEC TREE MSE:  {}".format(str(dectree_mse)))


def convert(data):
    number = preprocessing.LabelEncoder()
    cats = ['trim', 'isOneOwner', 'color', 'fuel', 'region', 'soundSystem', 'wheelType']
    for c in cats:
        data[c] = number.fit_transform(data[c])
    return data


def q17(data):
    """
    """
    y = data['price']
    x = data.drop(columns=["price"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)
    # train = x_train
    # train['price'] = y_train
    x_train = convert(x_train)
    x_test = convert(x_test)
    
    mse_scores = []
    for leaf in range(2,1000):
        tree = DecisionTreeRegressor(max_leaf_nodes=leaf)
        scores = cross_val_score(tree, x_train, y_train, cv=5, scoring='neg_mean_squared_error')
        # print("Degree: {}   scores: ".format(degree), end = "")
        print("leaves: {}: ".format(leaf), end = "")
        # print(scores, end="")
        print("    mse: " + str(-1 * scores.mean()))
        mse_scores.append(-1 * scores.mean())
    
    min_val = 100000000000
    for leaf, mse in zip(range(2,1000), mse_scores):
        if mse < min_val:
            min_val = mse
            num = leaf
        print("leaf = {}, mse = {}".format(str(leaf), str(mse)))
    print("Best Number of Leaf Nodes:  {},  MSE:  {}".format(num, min_val))
    
    plt.plot(list(range(2,1000)), mse_scores)
    plt.xlabel("leaves")
    plt.ylabel("mse")
    plt.savefig('tree_mse_allvar' + '.png', dpi=300)
    plt.close()
    
    
    tree = DecisionTreeRegressor(max_leaf_nodes=num)
    tree.fit(x_train, y_train)
    
    # dectree mse
    dectree_mse = mean_squared_error(y_test, tree.predict(x_test))
    print("DEC TREE MSE:  {}".format(str(dectree_mse)))


def convert_one_hot(data):
    number = preprocessing.OneHotEncoder()
    cats = ['trim', 'isOneOwner', 'color', 'fuel', 'region', 'soundSystem', 'wheelType']
    for c in cats:
        data[c] = number.fit_transform(data[c])
    return data


def bonus(data):
    """
    SBS
    """
    y = data['price']
    x = data.drop(columns=["price"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=3)
    # train = x_train
    # train['price'] = y_train
    x_train = convert(x_train)
    x_test = convert(x_test)
    
    covars = ['trim', 'isOneOwner', 'mileage', 'year', 'color', 'displacement', 'fuel', 'region', 'soundSystem', 'wheelType']
    cols = []

    while len(covars) > 1:
        min_score = 1000000000000
        for cov in covars:
            temp_covs = list(set(covars) - set([cov]))
            tree = DecisionTreeRegressor()
            scores = cross_val_score(tree, x_train[temp_covs], y_train, cv=5, scoring='neg_mean_squared_error')
            # print("score:  {}  var:  {}".format(str(scores.mean()), cov))
            if (-1 * scores.mean()) < min_score:
                min_score = (-1 * scores.mean())
                maxcov = cov
        covars = list(set(covars) - set([maxcov]))
        print("Removed: " + maxcov + "     MSE: " +  str(min_score))
    tree = DecisionTreeRegressor()
    scores = cross_val_score(tree, x_train[covars], y_train, cv=5, scoring='neg_mean_squared_error')
    print("Final feature: {}     MSE: ".format(covars[0]) +  str((-1) * scores.mean()))
    
    covars = ['trim', 'isOneOwner', 'mileage', 'year', 'color', 'displacement', 'fuel', 'region', 'soundSystem', 'wheelType']
    cols = []
    
    while covars:
        min_score = 1000000000000
        for cov in covars:
            temp_covs = cols + [cov]
            tree = DecisionTreeRegressor()
            scores = cross_val_score(tree, x_train[temp_covs], y_train, cv=5, scoring='neg_mean_squared_error')
            # print("score:  {}  var:  {}".format(str(scores.mean()), cov))
            if (-1 * scores.mean()) < min_score:
                min_score = (-1 * scores.mean())
                maxcov = cov
        cols.append(maxcov)
        covars = list(set(covars) - set([maxcov]))
        print("added: " + maxcov + "     MSE: " +  str(min_score))
        # print(cols)
    
    


def get_data():
    data = pd.read_csv("https://raw.githubusercontent.com/ChicagoBoothML/MLClassData/master/UsedCars/UsedCars.csv")
    return data
    

if __name__ == '__main__':
    data = get_data()
    q1(data)
    # q15(data)
    # q16(data)
    # q17(data)
    # bonus(data)
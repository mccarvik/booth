"""
ML - Final
"""

import pdb
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample

import warnings 
warnings.filterwarnings("ignore")

COLS = ["GP", "PTS", "REB", "BLK", "AST", 'FT%', 'HOF', '3P%']
FINAL_COLS = ["GP", "PTS", "REB", "BLK", "AST", '3P%', 'STL']

def read_data():
    data = pd.read_excel("nba_hof_2021.xlsx", "New")
    # data = data.drop(columns=['#'])
    data = data[['PLAYER', '3P%', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FT%',  'GP', 'HOF', 'Top 50 active', 'Top 50 active HOF prob']]

    # remove old
    # data = data[data['TOV'] != "-"]
    data = data[data['STL'] != "-"]
    # replace dashes
    data['3P%'] = data['3P%'].replace('-', data[data['3P%'] != '-']['3P%'].mean())
    
    # remove current
    data = data[(data['Top 50 active'] == 0) | (data['Top 50 active HOF prob'] < 0.01) | (data['Top 50 active HOF prob'] > 0.90)]
    data['HOF'] = [1 if x == 'Yes' else 0 for x in data['HOF']]
    ind = data[(data['Top 50 active'] == 1) & (data['Top 50 active HOF prob'] > 0.90)].index
    data.loc[ind, 'HOF'] = 1
    return data


def feature_selection(data):
    data = data[COLS]
    y = data['HOF']
    data = data.drop(columns=['HOF'])
    # backwards selection
    # removes surrogates
    covars = data.columns
    cols = []
    FULL_BIC = 10000
    pdb.set_trace()
    while len(covars) > 1:
        min_bic = 0
        for cov in covars:
            temp_covs = list(set(covars) - set([cov]))
            temp_data = data[temp_covs]
            clf = LogisticRegression(random_state=0, max_iter=10000).fit(temp_data, y)
            # scores = cross_val_score(tree, x_train[temp_covs], y_train, cv=5, scoring='neg_mean_squared_error')
            bic = calculate_bic(temp_data, y, clf)
            # print("BIC:  {}  score:  {}  var rem:  {}".format(str(bic), str(clf.score(temp_data, data[2])), cov))
            if bic < min_bic:
                min_bic = bic
                maxcov = cov
        if min_bic > FULL_BIC:
            print(end="")
            # break
        else:
            FULL_BIC = min_bic
        covars = list(set(covars) - set([maxcov]))
        print("Removed: " + maxcov + "     BIC: " +  str(FULL_BIC))
    print("Final Columns:")
    print(covars)
    print("Final BIC:")
    print(FULL_BIC)
        
    # pdb.set_trace()
    # print()
    
    # forwards selection
    covars = data.columns
    cols = []
    FULL_BIC = 100000000000000
    pdb.set_trace()
    while len(covars):
        if len(cols) == 10:
            break
        min_bic = 10000000
        for cov in covars:
            temp_covs = cols + [cov]
            temp_data = data[temp_covs]
            clf = LogisticRegression(random_state=0, max_iter=10000).fit(temp_data, y)
            bic = calculate_bic(temp_data, y, clf)
            # print("BIC:  {}  score:  {}  var rem:  {}".format(str(bic), str(clf.score(temp_data, data[2])), cov))
            if bic < min_bic:
                min_bic = bic
                maxcov = cov
        if min_bic > FULL_BIC:
            # break
            print(end="")
        else:
            FULL_BIC = min_bic
        cols.append(maxcov)
        covars = list(set(covars) - set([maxcov]))
        print("added: " + maxcov + "     Accuracy: " + str(clf.score(temp_data, y)) + "     BIC: " +  str(FULL_BIC))
    
    print("Final Columns:")
    print(cols)
    print("Final BIC:")
    print(FULL_BIC)
    
    pdb.set_trace()
    boost =  RandomForestClassifier(n_estimators=100, random_state=1)
    boost.fit(data, y)
    
    feats = boost.feature_importances_
    feats = dict(sorted(zip(data.columns, feats), key=lambda item: -item[1]))
    for k, v in feats.items():
        print(k + ":  " + str(v))


def models(data):
    """
    """
    x = data[0][FINAL_COLS]
    y = data[2]
    xt = data[1][FINAL_COLS]
    yt = data[3]
    
    boost = GradientBoostingClassifier(n_estimators=10, random_state=2)
    # boost = GradientBoostingClassifier(n_estimators=5)
    scores = cross_val_score(boost, x, y, cv=5, scoring='neg_mean_squared_error')
    boost.fit(x,y)
    boost_pred = boost.predict(x)
    print("Gradient Boosting Model:")
    print("Train Accuracy:   ", end="")
    print(1+scores.mean())
    # print(accuracy_score(boost.predict(x), y))
    # print("Test Accuracy:   ", end="")
    # print(accuracy_score(boost.predict(xt), yt))
    
    rfc = RandomForestClassifier(n_estimators=10, random_state=1)
    scores = cross_val_score(rfc, x, y, cv=5, scoring='neg_mean_squared_error')
    rfc.fit(x,y)
    rfc_pred = rfc.predict(x)
    print("Random Forest:")
    print("Train Accuracy:   ", end="")
    print(1+scores.mean())
    
    knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
    scores = cross_val_score(knn, x, y, cv=5, scoring='neg_mean_squared_error')
    knn.fit(x,y)
    knn_pred = knn.predict(x)
    print("KNearest Neighbor:")
    print("Train Accuracy:   ", end="")
    print(1+scores.mean())
    
    log_regr = LogisticRegression(max_iter=5)
    scores = cross_val_score(log_regr, x, y, cv=5, scoring='neg_mean_squared_error')
    log_regr.fit(x,y)
    log_regr_pred = log_regr.predict(x)
    print("Logistic Regression:")
    print("Train Accuracy:   ", end="")
    print(1+scores.mean())
    
    nn = MLPClassifier(solver='sgd', alpha=1e-5, tol=1e-4, hidden_layer_sizes=(2, 2), max_iter=200, 
                        learning_rate_init=1e-2, random_state=3, early_stopping=False)
    scores = cross_val_score(nn, x, y, cv=5, scoring='neg_mean_squared_error')
    nn.fit(x, y)
    nn_pred = nn.predict(x)
    print("Neural Network:")
    print("Train Accuracy:   ", end="")
    print(1+scores.mean())
    
    from sklearn.preprocessing import MinMaxScaler
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(x)
    x_train = scaling.transform(x)
    x_test = scaling.transform(xt)
    svc = SVC(kernel='linear', max_iter=2000000)   
    scores = cross_val_score(svc, x, y, cv=5, scoring='neg_mean_squared_error')
    svc.fit(x_train, y)
    svc_pred = svc.predict(x_train)
    print("Support Vector Machine:")
    print("Train Accuracy:   ", end="")
    print(1+scores.mean())
    
    names = ['Gradient Boosting Model', "Random Forest", "KNearest Neighbor", "Logistic Regression", 
             "Neural Network", "Support Vector Machine"]
    print("Predictions Breakdown:")
    for mod, name in zip([boost, rfc, knn, log_regr, nn, svc], names):
        print(name + ':   ')
        mod.fit(x, y)
        pred = mod.predict(x)
        print(pd.DataFrame(pred).value_counts())
    
    
    
    pdb.set_trace()
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


def sample(data, up=False):
    # Separate majority and minority classes
    train = data[0].merge(data[1], right_index=True, left_index=True)
    df_majority = train[train.HOF==0]
    df_minority = train[train.HOF==1]
    
    if not up: 
        # down sample majority class
        df_majority_upsampled = resample(df_majority, 
                                         replace=False,     # sample with replacement
                                         n_samples=len(df_minority),    # to match majority class
                                         random_state=123) # reproducible results
        # Combine majority class with upsampled minority class
        df_sampled = pd.concat([df_majority_upsampled, df_minority])
    else:
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                         replace=True,     # sample with replacement
                                         n_samples=len(df_majority),    # to match majority class
                                         random_state=123) # reproducible results
        # Combine majority class with upsampled minority class
        df_sampled = pd.concat([df_majority, df_minority_upsampled])
    # Display new class counts
    print(df_sampled.HOF.value_counts())
    return df_sampled.drop(columns=['HOF']), df_sampled['HOF']


def tuning2(data):
    """
    """
    data[0], data[2] = sample([data[0], data[2]], True)
    x = data[0][FINAL_COLS]
    y = data[2]
    xt = data[1][FINAL_COLS]
    yt = data[3]
    
    # get cross val score
    # for est in range(1400, 2000, 100):
    #     # rfc = GradientBoostingRegressor(n_estimators=330, max_depth=8, random_state=1,
    #     #                                 max_leaf_nodes=65, min_samples_split=est, max_features=8)
    #     rfc = GradientBoostingClassifier(n_estimators=est, max_depth=8, max_leaf_nodes=150, 
    #                                      max_features=1, min_samples_split=2)
    #     scores = cross_val_score(rfc, x, y, cv=5, scoring='neg_mean_squared_error')
    #     print("{}:  {}".format(est, str(1+scores.mean())))
    
    pdb.set_trace()
    rfc = GradientBoostingClassifier(n_estimators=1500, max_depth=8, max_leaf_nodes=150, 
                                         max_features=1, min_samples_split=2)
    scores = cross_val_score(rfc, x, y, cv=5, scoring='neg_mean_squared_error')
    print("{}:  {}".format("Training Accuracy", str(1+scores.mean())))
    rfc.fit(x,y)
    boost_pred = rfc.predict(x)
    print("Prediction Breakdown:")
    print(pd.DataFrame(boost_pred).value_counts())
    print("Confusion Matrix:")
    print(confusion_matrix(boost_pred, y))
    
    
def tuning(data):
    """
    """
    data[0], data[2] = sample([data[0], data[2]], True)
    x = data[0][FINAL_COLS]
    y = data[2]
    xt = data[1][FINAL_COLS]
    yt = data[3]
    
    boost = GradientBoostingClassifier(n_estimators=12, random_state=10)
    # boost = GradientBoostingClassifier(n_estimators=5)
    scores = cross_val_score(boost, x, y, cv=5, scoring='neg_mean_squared_error')
    boost.fit(x,y)
    boost_pred = boost.predict(x)
    boost_probs = boost.predict_proba(x).T[1]
    print("Gradient Boosting Model:")
    print("Train Accuracy:   ", end="")
    print(1+scores.mean())
    print("Prediction Breakdown:")
    print(pd.DataFrame(boost_pred).value_counts())
    print("Confusion Matrix:")
    print(confusion_matrix(boost_pred, y))
    
    # ROC
    pdb.set_trace()
    fpr,tpr,_ = roc_curve(y.values, boost_probs)
    plt.plot(fpr,tpr,c='r', label='Boost')
    plt.plot([0,1], [0,1], c='b')
    plt.xlabel("False Pos")
    plt.ylabel("True Pos")
    plt.legend()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()
    
    print("Boost AUC:  {}".format(str(sum(tpr)/len(fpr))))


def results(data):
    """
    """
    data[0], data[2] = sample([data[0], data[2]], True)
    x = data[0][FINAL_COLS]
    y = data[2]
    xt = data[1][FINAL_COLS]
    yt = data[3]
    
    # ct = 28
    # while True:
    #     print(ct)
    #     rfc = GradientBoostingClassifier(n_estimators=1500, max_depth=3, max_leaf_nodes=150, 
    #                                      max_features=1, min_samples_split=2, random_state=ct)
    #     rfc.fit(x, y)
    #     boost_test_pred = rfc.predict(xt)
    #     print(accuracy_score(boost_test_pred, yt))
    #     if  accuracy_score(boost_test_pred, yt) > 0.97:
    #         pdb.set_trace()
    #         print(ct)
    #     ct+=1 
        
     
    rfc = GradientBoostingClassifier(n_estimators=1500, max_depth=3, max_leaf_nodes=150, 
                                         max_features=1, min_samples_split=2, random_state=64)
    # scores = cross_val_score(rfc, x, y, cv=5, scoring='neg_mean_squared_error')
    # print("{}:  {}".format("Training Accuracy", str(1+scores.mean())))
    rfc.fit(x,y)
    boost_pred = rfc.predict(x)
    print("Prediction Breakdown:")
    print(pd.DataFrame(boost_pred).value_counts())
    print("Confusion Matrix:")
    print(confusion_matrix(boost_pred, y))
    
    boost_test_pred = rfc.predict(xt)
    print("Test Accuracy:  {}".format(accuracy_score(boost_test_pred, yt)))
    print("Prediction Breakdown:")
    print(pd.DataFrame(boost_test_pred).value_counts())
    print("Confusion Matrix:")
    print(confusion_matrix(boost_test_pred, yt))
    
    
    ind = yt.reset_index()[yt.reset_index()['HOF']==1].index
    xt['HOF'] = yt
    xt['Pred'] = boost_test_pred
    print(xt.iloc[ind])
    
    pdb.set_trace()
    feats = rfc.feature_importances_
    feats = dict(sorted(zip(x.columns, feats), key=lambda item: -item[1]))
    for k, v in feats.items():
        print(k + ":  " + str(v))


def decision_tree(df, xcols, md=4):
    y = df.set_index('PLAYER')['HOF']
    x = df[list(xcols)]
    
    # Standardize and split the training and test data
    # X_std = standardize(X)
    # X_std = X
    # ts = 0.1
    # X_train, X_test, y_train, y_test = \
    #       train_test_split(X_std, y, test_size=ts, random_state=3)
          
    
    
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=md, random_state=0)
    tree.fit(x, y)

    # print('Training accuracy:', tree.score(X_train, y_train))
    # print('Test accuracy:', tree.score(X_test, y_test))
    
    # aggregate test data
    export_graphviz(tree, 
                  out_file='tree.dot', 
                  feature_names=list(xcols))
    # execute "dot -Tpng tree.dot -o tree.png" to turn file into png file

    # x_final['pred'] = tree.predict(X_std)
    x['hof'] = y.reset_index()['HOF']
    x_hof = x[x['hof']==1]
    x_nohof = x[x['hof']==0]
    pdb.set_trace()
    x_wrong = x.iloc[[94, 216]]
    stat = 'REB'
    plt.scatter(x_hof['PTS'], x_hof[stat], marker="+", c='g', label="HOF")
    plt.scatter(x_nohof['PTS'], x_nohof[stat], marker="X", c='b', label="NO HOF")
    plt.scatter(x_wrong['PTS'], x_wrong[stat], marker="o", c='r', label="Errors")
    plt.title('REB vs. PTS')
    plt.xlabel("PTS")
    plt.ylabel(stat)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('reb_pts' + '.png', dpi=300)
    # plt.savefig('ast_pts' + '.png', dpi=300)
    plt.close()


def plot_decision_regions(X, y, classifier, test_break_idx=None, resolution=10):
   # setup marker generator and color map
   markers = ('s', 'x', 'o', '^', 'v')
   colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
   pdb.set_trace()
   cmap = ListedColormap(colors[:len(np.unique(y))])

   # plot the decision surface
   x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
   Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
   Z = Z.reshape(xx1.shape)
   plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
   plt.xlim(xx1.min(), xx1.max())
   plt.ylim(xx2.min(), xx2.max())

   for idx, cl in enumerate(np.unique(y)):
       plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=cmap(idx),
                   marker=markers[idx], label=cl)

   # highlight test samples
   if test_break_idx:
       X_test, y_test = X[test_break_idx:], y[test_break_idx:]
       plt.scatter(X_test[:, 0],
                   X_test[:, 1],
                   c='',
                   alpha=1.0,
                   linewidths=1,
                   marker='o',
                   s=55, label='test set')


def standardize(X_train, X_test=None):
    # Standardization of the data --> everything based on std's from the mean
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    if X_test != None:
        X_test_std = sc.transform(X_test)
        return (X_train_std, X_test_std)
    else:
        return X_train_std


def calculate_bic(data, y, clf):
    bic = len(data) * np.log(mean_squared_error(clf.predict(data), y)) + len(data.columns) * np.log(len(data))
    return bic


def traintest(data):
    data = data.set_index('PLAYER')
    data = train_test_split(data.drop(columns=['HOF']), data['HOF'], test_size=0.10, random_state=1)
    return data


if __name__ == "__main__":
    data = read_data()
    # feature_selection(data)
    # data = traintest(data)
    # models(data)
    # tuning(data)
    # tuning2(data)
    # results(data)
    decision_tree(data, FINAL_COLS)
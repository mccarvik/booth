"""
HW1 AI
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
from sklearn.model_selection import train_test_split


def read_data():
    data = pd.read_excel("nba hof.xlsx")
    data = data.set_index("PLAYER")
    # remove any old or current players
    data = data[data['current_old'] == 0]
    return data



def decision_tree(df, xcols, md=4):
    y = df['HOF']
    X = df[list(xcols)]
    
    # Standardize and split the training and test data
    # X_std = standardize(X)
    X_std = X
    ts = 0.1
    X_train, X_test, y_train, y_test = \
          train_test_split(X_std, y, test_size=ts, random_state=3)
          
    
    
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=md, random_state=0)
    tree.fit(X_train, y_train)

    print('Training accuracy:', tree.score(X_train, y_train))
    print('Test accuracy:', tree.score(X_test, y_test))
    
    # aggregate test data
    pdb.set_trace()
    xt_final = X_test
    xt_final['pred'] = tree.predict(X_test)
    xt_final['actual'] = y_test
    xt_final.to_csv("test_set.csv")
    
    # plot_decision_regions(X_std.values, y.values, classifier=tree)
    # plt.title('Decision Tree')
    # plt.xlabel(list(X.columns)[0])
    # plt.ylabel(list(X.columns)[1])
    # plt.legend(loc='upper left')
    # plt.tight_layout()
    # plt.savefig('dec_tree' + '.png', dpi=300)
    # plt.close()
    
    export_graphviz(tree, 
                  out_file='tree.dot', 
                  feature_names=list(xcols))
    # execute "dot -Tpng tree.dot -o tree.png" to turn file into png file

    x_final = X_std
    x_final['pred'] = tree.predict(X_std)
    x_final['actual'] = y
    x_final['correct'] = x_final['pred'] == x_final['actual']
    x_correct = x_final[x_final['correct']]
    x_false = x_final[~x_final['correct']]
    plt.scatter(x_correct['PTS'], x_correct['REB'], marker=">", label="Correct")
    plt.scatter(x_false['PTS'], x_false['REB'], marker="+", label="Incorrect")
    plt.title('REB vs. PTS')
    plt.xlabel("PTS")
    plt.ylabel("REB")
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('reb_pts' + '.png', dpi=300)
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




if __name__ == "__main__":
    data = read_data()
    # cols = ["GP", "PTS", "REB", "BLK", "AST", "STL", "TOV"]
    cols = ["PTS", "REB", "BLK", "AST", "STL"]
    decision_tree(data, cols)
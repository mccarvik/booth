
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

LABELS = {
    1: "virginica",
    2: "versicolor"
}


def plot_decision_regions(X, y, classifier, train_idx=None, resolution=0.02):
    
    # setup marker generator and color map
    markers = ('+', 'x', 'o', '^', 'v')
    colors = ('lightblue', 'darkgreen', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    # x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    # xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    #                       np.arange(x2_min, x2_max, resolution))
    # pdb.set_trace()
    # Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # Z = Z.reshape(xx1.shape)
    # plt.contourf(xx1, xx2, Z, alpha=0.2, cmap=cmap)
    # plt.xlim(xx1.min(), xx1.max())
    # plt.ylim(xx2.min(), xx2.max())

    # for idx, cl in enumerate(np.unique(y)):
    #     plt.scatter(x=X[y == cl, 0], 
    #                 y=X[y == cl, 1],
    #                 alpha=0.6, 
    #                 c=cmap(idx),
    #                 edgecolor='black',
    #                 marker=markers[idx])

    # highlight test samples
    # pdb.set_trace()
    if train_idx:
        # plot all samples
        X_test, y_test = X[:train_idx], y[:train_idx]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='red',
                    linewidths=1,
                    marker='o',
                    s=55, label='train set')
    
    addrows = True
    if addrows:
        mis_ind = abs(classifier.predict(X) - y)
        ct = 0
        errs = []
        for i in mis_ind:
            if i > 0:
                errs.append(ct)
            ct+=1

        # plot all samples
        X_test, y_test = np.array(X)[errs], y[errs]
        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=10.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=100, label='errors')
        
        
        # # plot all samples
        # X_test, y_test = X[-20:], y[-20:]
        # plt.scatter(X_test[:, 0],
        #             X_test[:, 1],
        #             c='',
        #             alpha=1.0,
        #             edgecolor='red',
        #             linewidths=1,
        #             marker='o',
        #             s=55, label='new rows')


    # setup marker generator and color map
    markers = ('+', 'x', 'o', '^', 'v')
    colors = ('darkblue', 'darkgreen', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    # x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x1_min, x1_max = -3.31895381717002, 3.456235883146482
    # x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x2_min, x2_max = -2.602934200868721, 2.998211469766752
    
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel(), [0]*len(xx1.ravel())]).T)
    # Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    # Z = classifier.predict(np.array([xx1.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    X = np.array([[x[0], x[1]] for x in X])
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=LABELS[cl])


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

# only 1 and 2
y = y[50:]
X = X[50:]


for i in range(1,100,1):
    print(i)
    # new columns
    new_col = np.random.rand(1, 100) * 10 - 5
    
    
    # new rows
    # new_y = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    # new_x = [[5.2, 1.8], [5.3, 1.9], [5.4, 2.0], [5.6, 1.8], [5.4, 1.9],
    #          [5.2, 1.9], [5.5, 1.9], [5.4, 2.1], [5.5, 1.8], [5.3, 2.0],
    #          [5.0, 1.9], [5.0, 1.6], [5.1, 1.8], [5.3, 1.7], [5.3, 1.6],
    #          [5.0, 1.7], [4.9, 1.5], [5.0, 1.8], [5.2, 1.5], [5.3, 1.5]]
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.000001, random_state=i)
    
    # new rows
    # y_train = np.concatenate([y_train, new_y]) 
    # X_train = np.concatenate([X_train, new_x])
    
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    
    # Add new col
    X_train_std = [[x[0], x[1], y] for x,y in zip(X_train_std, new_col[0])]
    
    
    lr = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')
    
    # remove columns
    # Xc = [x[0] for x in X_train_std] 
    # Xct = np.array([x[0] for x in X_test_std])
    
    # original
    lr.fit(X_train_std, y_train)
    # remove a column
    # lr.fit(np.array(Xc).reshape(-1,1), y_train)
    
    # plot_decision_regions(X_combined_std, y_combined, classifier=lr, train_idx=20)
    # plot_decision_regions(X_combined_std, y_combined, classifier=lr)
    plot_decision_regions(X_train_std, y_train, classifier=lr)
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('temp.png', dpi=300)
    plt.close()
    
    # print(accuracy_score(lr.predict(np.array(Xc).reshape(-1,1)), y_train))
    print(accuracy_score(lr.predict(X_train_std), y_train))
    # print(accuracy_score(lr.predict(X_test_std), y_test))
    
    if accuracy_score(lr.predict(X_train_std), y_train) < 0.94:
        break

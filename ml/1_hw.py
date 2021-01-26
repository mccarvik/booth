"""
HW1 ML
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


np.random.seed(42)

def get_data():
    # part 1 - create data
    # y = 1.8x + 2 + e
    x_train = np.random.rand(100)*2-1
    e = np.random.normal(0, 0.1, 100)
    y_train = 1.8 * x_train + 2 + e
    x_test = np.random.rand(10000)*2-1
    e = np.random.normal(0, 0.1, 10000)
    y_test = 1.8 * x_test + 2 + e 
    
    # part 2 - draw a scatter plot
    # x_vals = np.linspace(-1,1,100)
    # line = x_vals * 1.8 + 2
    # plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    # plt.plot(x_vals, line, c="k")
    # plt.savefig('train' + '.png', dpi=300)
    # plt.close()
    # plt.scatter(x_test, y_test,  s=5, facecolors='none', edgecolors='r')
    # plt.plot(x_vals, line, c="k")
    # plt.savefig('test' + '.png', dpi=300)
    # plt.close()

    # part 3 - add regression line
    # df = pd.DataFrame()
    # df['x'] = x_train
    # df['y'] = y_train
    # reg = sm.ols(formula="y ~ x", data=df).fit()
    # beta = reg.params[1]
    # alpha = reg.params['Intercept']
    # lin_mse = reg.mse_total
    # plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    # x_vals = np.linspace(-1,1,100)
    # line = x_vals * beta + alpha
    # plt.plot(x_vals, line, "--", c="b")
    # x_vals = np.linspace(-1,1,100)
    # line = x_vals * 1.8 + 2
    # plt.plot(x_vals, line, c="k")
    # plt.savefig('train_ols' + '.png', dpi=300)
    # plt.close()
    
    # part 4 - knn
    # k = 2
    # knn = KNeighborsRegressor(n_neighbors=2, p=2, metric='minkowski')
    # knn.fit(x_train.reshape(-1, 1), y_train)
    # x_vals = np.linspace(-1,1,100)
    # y = knn.predict(x_vals.reshape(-1, 1))
    # line = x_vals * 1.8 + 2
    # plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    # plt.plot(x_vals, line, c="k")
    # plt.plot(x_vals, y, "--", c="b")
    # plt.title('KNN=2')
    # plt.savefig('train_knn_2_' + '.png', dpi=300)
    # plt.close()
    
    # k = 12
    # knn = KNeighborsRegressor(n_neighbors=12, p=2, metric='minkowski')
    # knn.fit(x_train.reshape(-1, 1), y_train)
    # y = knn.predict(x_vals.reshape(-1, 1))
    # line = x_vals * 1.8 + 2
    # plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    # plt.plot(x_vals, line, c="k")
    # plt.plot(x_vals, y, "--", c="b")
    # plt.title('KNN=12')
    # plt.savefig('train_knn_12_' + '.png', dpi=300)
    # plt.close()
    
    
    # part 5 - Mean Squared Error
    knn_mse = []
    log_k = []
    for k in range(2,16):
        knn = KNeighborsRegressor(n_neighbors=k, p=2, metric='minkowski')
        knn.fit(x_train.reshape(-1, 1), y_train)
        y_pred = knn.predict(x_test.reshape(-1, 1))
        mse = mean_squared_error(y_test, y_pred)
        knn_mse.append(mse)
        log_k.append(np.log(1/k))
    
    for k, mse in zip(range(2,16), knn_mse):
        print("k = {}, mse = {}".format(str(k), str(mse)))
    
    
    df = pd.DataFrame()
    df['x'] = x_train
    df['y'] = y_train
    reg = sm.ols(formula="y ~ x", data=df).fit()
    dft = pd.DataFrame()
    dft['x'] = x_test
    y_pred = reg.predict(dft)
    lin_mse = mean_squared_error(y_test, y_pred)
    
    plt.plot(log_k, knn_mse, "-", c="g", label="knn")
    plt.plot(log_k, [lin_mse]*len(log_k), "--", c="k", label="lin")
    plt.xlabel("LOG(1/k)")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('MSE_linear' + '.png', dpi=300)
    plt.close()


# Second function
def get_data2():
    # part 1 - create data
    # y = tanh(1.1 * x) + 2 + e
    x_train = np.random.rand(100)*2-1
    e = np.random.normal(0, 0.1, 100)
    y_train = np.tanh(1.1 * x_train) + 2 + e
    x_test = np.random.rand(10000)*2-1
    e = np.random.normal(0, 0.1, 10000)
    y_test = np.tanh(1.1 * x_test) + 2 + e
    
    # part 2 - draw a scatter plot
    x_vals = np.linspace(-1,1,100)
    line = np.tanh(1.1 * x_vals) + 2
    plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    plt.plot(x_vals, line, c="k")
    plt.savefig('train_tanh' + '.png', dpi=300)
    plt.close()
    plt.scatter(x_test, y_test,  s=5, facecolors='none', edgecolors='r')
    plt.plot(x_vals, line, c="k")
    plt.savefig('test_tanh' + '.png', dpi=300)
    plt.close()

    # part 3 - add regression line
    df = pd.DataFrame()
    df['x'] = x_train
    df['y'] = y_train
    reg = sm.ols(formula="y ~ x", data=df).fit()
    beta = reg.params[1]
    alpha = reg.params['Intercept']
    lin_mse = reg.mse_total
    plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    x_vals = np.linspace(-1,1,100)
    line = x_vals * beta + alpha
    plt.plot(x_vals, line, "--", c="b")
    x_vals = np.linspace(-1,1,100)
    line = np.tanh(1.1 * x_vals) + 2
    plt.plot(x_vals, line, c="k")
    plt.savefig('train_ols_tanh' + '.png', dpi=300)
    plt.close()
    
    # part 4 - knn
    # k = 2
    knn = KNeighborsRegressor(n_neighbors=2, p=2, metric='minkowski')
    knn.fit(x_train.reshape(-1, 1), y_train)
    x_vals = np.linspace(-1,1,100)
    y = knn.predict(x_vals.reshape(-1, 1))
    line = np.tanh(1.1 * x_vals) + 2
    plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    plt.plot(x_vals, line, c="k")
    plt.plot(x_vals, y, "--", c="b")
    plt.title('KNN=2')
    plt.savefig('train_knn_2_tanh' + '.png', dpi=300)
    plt.close()
    
    # k = 12
    knn = KNeighborsRegressor(n_neighbors=12, p=2, metric='minkowski')
    knn.fit(x_train.reshape(-1, 1), y_train)
    y = knn.predict(x_vals.reshape(-1, 1))
    line = np.tanh(1.1 * x_vals) + 2
    plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    plt.plot(x_vals, line, c="k")
    plt.plot(x_vals, y, "--", c="b")
    plt.title('KNN=12')
    plt.savefig('train_knn_12_tanh' + '.png', dpi=300)
    plt.close()
    
    
    # part 5 - Mean Squared Error
    knn_mse = []
    log_k = []
    for k in range(2,16):
        knn = KNeighborsRegressor(n_neighbors=k, p=2, metric='minkowski')
        knn.fit(x_train.reshape(-1, 1), y_train)
        y_pred = knn.predict(x_test.reshape(-1, 1))
        mse = mean_squared_error(y_test, y_pred)
        knn_mse.append(mse)
        log_k.append(np.log(1/k))
        
    for k, mse in zip(range(2,16), knn_mse):
        print("k = {}, mse = {}".format(str(k), str(mse)))
    
    df = pd.DataFrame()
    df['x'] = x_train
    df['y'] = y_train
    reg = sm.ols(formula="y ~ x", data=df).fit()
    dft = pd.DataFrame()
    dft['x'] = x_test
    y_pred = reg.predict(dft)
    lin_mse = mean_squared_error(y_test, y_pred)
    
    plt.plot(log_k, knn_mse, "-", c="g", label="knn")
    plt.plot(log_k, [lin_mse]*len(log_k), "--", c="k", label="lin")
    plt.xlabel("LOG(1/k)")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('MSE_linear_tanh' + '.png', dpi=300)
    plt.close()


# Third function
def get_data3():
    # part 1 - create data
    # y = sin(2 * x) + 2
    x_train = np.random.rand(100)*2-1
    e = np.random.normal(0, 0.1, 100)
    y_train = np.sin(2 * x_train) + 2 + e
    x_test = np.random.rand(10000)*2-1
    e = np.random.normal(0, 0.1, 10000)
    y_test = np.sin(2 * x_test) + 2 + e
    
    # part 2 - draw a scatter plot
    x_vals = np.linspace(-1,1,100)
    line = np.sin(2 * x_vals) + 2
    plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    plt.plot(x_vals, line, c="k")
    plt.savefig('train_sin' + '.png', dpi=300)
    plt.close()
    plt.scatter(x_test, y_test,  s=5, facecolors='none', edgecolors='r')
    plt.plot(x_vals, line, c="k")
    plt.savefig('test_sin' + '.png', dpi=300)
    plt.close()

    # part 3 - add regression line
    df = pd.DataFrame()
    df['x'] = x_train
    df['y'] = y_train
    reg = sm.ols(formula="y ~ x", data=df).fit()
    beta = reg.params[1]
    alpha = reg.params['Intercept']
    lin_mse = reg.mse_total
    plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    x_vals = np.linspace(-1,1,100)
    line = x_vals * beta + alpha
    plt.plot(x_vals, line, "--", c="b")
    x_vals = np.linspace(-1,1,100)
    line = np.sin(2 * x_vals) + 2
    plt.plot(x_vals, line, c="k")
    plt.savefig('train_ols_sin' + '.png', dpi=300)
    plt.close()
    
    # part 4 - knn
    # k = 2
    knn = KNeighborsRegressor(n_neighbors=2, p=2, metric='minkowski')
    knn.fit(x_train.reshape(-1, 1), y_train)
    x_vals = np.linspace(-1,1,100)
    y = knn.predict(x_vals.reshape(-1, 1))
    line = np.sin(2 * x_vals) + 2
    plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    plt.plot(x_vals, line, c="k")
    plt.plot(x_vals, y, "--", c="b")
    plt.title('KNN=2')
    plt.savefig('train_knn_2_sin' + '.png', dpi=300)
    plt.close()
    
    # k = 12
    knn = KNeighborsRegressor(n_neighbors=12, p=2, metric='minkowski')
    knn.fit(x_train.reshape(-1, 1), y_train)
    y = knn.predict(x_vals.reshape(-1, 1))
    line = np.sin(2 * x_vals) + 2
    plt.scatter(x_train, y_train, s=5, facecolors='none', edgecolors='r')
    plt.plot(x_vals, line, c="k")
    plt.plot(x_vals, y, "--", c="b")
    plt.title('KNN=12')
    plt.savefig('train_knn_12_sin' + '.png', dpi=300)
    plt.close()
    
    
    # part 5 - Mean Squared Error
    knn_mse = []
    log_k = []
    for k in range(2,16):
        knn = KNeighborsRegressor(n_neighbors=k, p=2, metric='minkowski')
        knn.fit(x_train.reshape(-1, 1), y_train)
        y_pred = knn.predict(x_test.reshape(-1, 1))
        mse = mean_squared_error(y_test, y_pred)
        knn_mse.append(mse)
        log_k.append(np.log(1/k))
        
    for k, mse in zip(range(2,16), knn_mse):
        print("k = {}, mse = {}".format(str(k), str(mse)))
    
    df = pd.DataFrame()
    df['x'] = x_train
    df['y'] = y_train
    reg = sm.ols(formula="y ~ x", data=df).fit()
    dft = pd.DataFrame()
    dft['x'] = x_test
    y_pred = reg.predict(dft)
    lin_mse = mean_squared_error(y_test, y_pred)
    
    plt.plot(log_k, knn_mse, "-", c="g", label="knn")
    plt.plot(log_k, [lin_mse]*len(log_k), "--", c="k", label="lin")
    plt.xlabel("LOG(1/k)")
    plt.ylabel("MSE")
    plt.legend()
    plt.savefig('MSE_linear_sin' + '.png', dpi=300)
    plt.close()



def get_data4():
    # part 1 - create data
    x_train = np.random.rand(1000)*2-1
    e = np.random.normal(0, 0.1, 1000)
    y_train = np.sin(2 * x_train) + 2 + e
    x_test = np.random.rand(10000)*2-1
    e = np.random.normal(0, 0.1, 10000)
    y_test = np.sin(2 * x_test) + 2 + e
    
    # for p 2 through 20
    x_test = [x_test]
    x_train = [x_train]
    for p in range(1,20):
        e = np.random.normal(0, 1, 1000)
        x_train.append(e)
        e = np.random.normal(0, 1, 10000)
        x_test.append(e)
        
        df = pd.DataFrame()
        dft = pd.DataFrame()
        df['x'] = x_train[0]
        df['y'] = y_train
        dft['x'] = x_test[0]
        form = "y ~ x"
        for i in range(p):
            strp = str(p)
            df["p" + strp] = x_train[p]
            dft["p" + strp] = x_test[p]
            form += " + p" + strp

        reg = sm.ols(formula=form, data=df).fit()
        y_pred = reg.predict(dft)
        lin_mse = mean_squared_error(y_test, y_pred)
        
        knn_mse = []
        log_k = []
        for k in range(2,15):
            knn = KNeighborsRegressor(n_neighbors=k, p=2, metric='minkowski')
            knn.fit(pd.DataFrame(x_train).T.values, y_train)
            y_pred = knn.predict(pd.DataFrame(x_test).T.values)
            mse = mean_squared_error(y_test, y_pred)
            knn_mse.append(mse)
            log_k.append(np.log(1/k))
        
        plt.plot(log_k, knn_mse, "-", c="g", label="knn")
        plt.plot(log_k, [lin_mse]*len(log_k), "--", c="k", label="lin")
        plt.xlabel("LOG(1/k)")
        plt.ylabel("MSE")
        plt.legend()
        pp = p + 1
        plt.title("MSE KNN vs LinReg, P = " + str(pp))
        plt.savefig('MSE_noise_' + str(pp) + '.png', dpi=300)
        plt.close()
        


if __name__ == '__main__':
    # get_data()
    # get_data2()
    get_data3()
    # get_data4()
    
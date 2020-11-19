"""
HW 5 Investments
"""

import pdb
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as sm
import statsmodels.api as stats
import scipy.stats as scipy
import warnings


def optchart(k, prem, lng=True, pc=True):
    x_vals = np.linspace(75,125,51)
    # x_vals = np.linspace(80,120,9)
    if pc:
        if lng:
            y_vals = np.array([0 if x < k else x-k for x in x_vals]) - prem
        else:
            y_vals = np.array([0 if x < k else -(x-k) for x in x_vals]) + prem
    else:
        if lng:
            y_vals = np.array([0 if x > k else k-x for x in x_vals]) - prem
        else:
            y_vals = np.array([0 if x > k else -(k-x) for x in x_vals]) + prem
    
    if pc:
        pcn = "Call"
    else:
        pcn = "Put"
    if lng:
        ln = "Long"
    else:
        ln = "Short"
    
    if lng:
        payoffs = y_vals + prem
    else:
        payoffs = y_vals - prem
    
    name = pcn + " " + ln + " strike=" + str(k) + "  premium=" + str(prem) 
    axes = plt.gca()
    axes.set_xlim([75, 125])
    axes.set_ylim([-50, 50])
    plt.plot(x_vals, y_vals, '-b', label="profit")
    plt.plot(x_vals, [0] * len(x_vals), '--r')
    plt.plot(x_vals, payoffs, '--b', label="payoff")
    plt.legend()
    plt.xlabel("Stock Price", fontsize=12)
    plt.ylabel("Payoff", fontsize=12)
    plt.title(name)
    plt.savefig(name + ".png",dpi=400)
    plt.close()
    return [x_vals, y_vals, payoffs]
    

opt1 = optchart(100, 6)
opt2 = optchart(90, 20)
opt3 = optchart(105, 8, False)
opt4 = optchart(95, 16, False)

df = pd.DataFrame([opt1[0], opt1[1], opt2[1], opt3[1], opt4[1]]).T.set_index(0)
dfp = pd.DataFrame([opt1[0], opt1[2], opt2[2], opt3[2], opt4[2]]).T.set_index(0)
df['sum'] = df.sum(axis=1)
dfp['sum'] = dfp.sum(axis=1)
df.columns = ['opt1', 'opt2', 'opt3', 'opt4', 'total']
dfp.columns = ['opt1', 'opt2', 'opt3', 'opt4', 'total']

axes = plt.gca()
axes.set_xlim([75, 125])
axes.set_ylim([-50, 50])
plt.plot(df.reset_index()[0], [0] * len(df.reset_index()[0]), '--r')
plt.plot(df.reset_index()[0], df['total'], '-b', label="profit")
plt.plot(df.reset_index()[0], dfp['total'], '--b', label="payoff")
plt.legend()
plt.xlabel("Stock Price", fontsize=12)
plt.ylabel("Payoff", fontsize=12)
plt.title("Total")
plt.savefig("total.png",dpi=400)
plt.close()

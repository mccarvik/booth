"""
Big Data Midterm
"""
import pdb
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import statsmodels.statsmodels.formula.api as sm
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
# import statsmodels.api as sm
from patsy import dmatrices


def ret_chart(dff):
    """
    return and vol charts
    """
    dff['Date'] =  pd.to_datetime(dff['Date'])
    dff.plot.line(x='Date', y='return')
    plt.xlabel("Date", fontsize=9)
    plt.ylabel("Return", fontsize=12)

    # myLocator = mticker.MultipleLocator(4)
    axes = plt.gca()
    axes.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=6))
    axes.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=50)
    plt.savefig("pngs/returns.png",dpi=400)
    plt.close()

    dff.plot.line(x='Date', y='log_sprd')
    plt.xlabel("Date", fontsize=9)
    plt.ylabel("Vol", fontsize=12)

    axes = plt.gca()
    axes.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=6))
    axes.xaxis.set_major_formatter(mpl.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=50)
    plt.savefig("pngs/vol.png",dpi=400)
    plt.close()


def prework():
    """
    Prework on the data
    """
    reddit = pd.read_csv("RedditNews.csv",header=None)
    reddit.columns = ['date', 'text']
    # subset = reddit[reddit.date=="7/1/16"]
    # print(subset)

    # read DJIA data
    djia = pd.read_csv("DJIA.csv")
    # print(djia.head)
    ndays = len(djia)
    # print(ndays)

    # Read the words
    words = pd.read_csv("WordsFinal.csv", header=None)
    words = words[0].tolist()
    # print(words[:7])

    # Need this to fix file
    # doc_word = pd.read_csv("WordFreqFinal.csv",header=None)
    # doc_word['0'] = doc_word.apply(lambda row: row[0].split(" ")[0], axis=1)
    # doc_word['1'] = doc_word.apply(lambda row: row[0].split(" ")[1], axis=1)
    # doc_word['2'] = doc_word.apply(lambda row: row[0].split(" ")[2], axis=1)
    doc_word = pd.read_csv("NewWordFreq.csv", header=None)
    # print(doc_word)

    # create a sparse matrix
    sparse_mat = csr_matrix((doc_word[2], (doc_word[0], doc_word[1])))

    # We select only words at occur at least 5 times
    sparse_df = pd.DataFrame.sparse.from_spmatrix(sparse_mat).sparse.to_dense()
    # first column androw were empty
    sparse_df = sparse_df.drop(columns=[0])
    sparse_df = sparse_df.drop(0)
    sparse_df.columns = words
    # must occur in more than 5 lists, must have less than ndays
    sparse_df = sparse_df[sparse_df.columns[sparse_df.sum()>5]]
    sparse_df = sparse_df[sparse_df.columns[sparse_df.sum()<ndays]]
    print(sparse_df)


def quest1():
    """
    Question 1
    """
    pdb.set_trace()
    import statsmodels.api as sm
    dat = sm.datasets.get_rdataset("Guerry", "HistData").data
    # Fit regression model (using the natural log of one of the regressors)
    results = smf.ols('Lottery ~ Literacy + np.log(Pop1831)', data=dat).fit()
    # Inspect the results
    print(results.summary())
    pdb.set_trace()

    start_df = pd.read_csv("prework.csv")
    djia = pd.read_csv("DJIA.csv")
    djia['lag1'] = djia.shift(-1)['Adj Close']
    djia['Ret'] = (djia['Adj Close'] - djia['lag1']) / djia['lag1']
    djia['log_sprd'] = np.log(djia['High']-djia['Low'])
    djia = djia.reindex(index=djia.index[::-1])
    # ret_chart(djia)
    start_df_bool = start_df>0
    start_df_bool = start_df_bool.astype(int)
    merged_df = djia.merge(start_df_bool, left_index=True, right_index=True)
    
    for col in start_df_bool.columns.tolist():
        pdb.set_trace()
        temp_df = merged_df[['Date', 'Ret', 'log_sprd', col]].dropna()
        reg = sm.ols(formula="Ret ~ {}".format(col), data=temp_df)
        reg.fit()
        
        
        beta = reg.params['ue_diff']
        alpha = reg.params['Intercept']
        
    print("O REG)")


if __name__ == "__main__":
    quest1()

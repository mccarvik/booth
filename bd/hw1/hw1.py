import pdb
import pandas as pd

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import statsmodels.api as stats
import scipy.stats as scipy
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")

review = pd.read_csv("./Review_subset.csv", sep=' ')
words = pd.read_csv("./words.csv", sep=' ')
# products = pd.read_csv("./products.csv", sep=' ', encoding = "utf-8")
word_freq = pd.read_csv("./word_freq.csv", sep=' ')


word_freq.columns = ["Review ID","Word ID","Times Word"]
print(review.shape)


# Make sparse matrix
pdb.set_trace()
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])
print(csr_matrix((data, (row, col)), shape=(3, 3)).toarray())

# spm = csr_matrix((word_freq['Times Word'].values, (word_freq['Review ID'].values, word_freq['Word ID'].values)), shape=(len(word_freq['Review ID'].values), len(word_freq['Word ID'].values)))

for word in words:
    pdb.set_trace()
    reg = sm.ols(formula="YES ~ SIZE + np.log(VAL)", data=beef).fit()

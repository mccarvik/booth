#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd

from math import sqrt

import scipy
from scipy.stats import pearsonr
import scipy.stats as stats
from scipy.stats import ttest_1samp
import statsmodels.formula.api as smf
from pandas import DataFrame
from scipy.stats import linregress


import statistics


import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.stattools as st
from statsmodels.formula.api import ols
from statsmodels.multivariate.manova import MANOVA
import warnings 
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import statsmodels.api as sm

#Sklearn
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge

plt.style.use('ggplot')


#  # 1.1.1
# 

# In[122]:


#Read in data
df = pd.read_csv("trump1.csv")


# In[136]:


df.loc[df['hour'] > 17, 'Hourbucket'] = 3
df.loc[df['hour'] <= 17, 'Hourbucket'] = 2
df.loc[df['hour'] <= 11, 'Hourbucket'] = 1
df.loc[df['hour'] <= 5, 'Hourbucket'] = 0 


# In[137]:


print(df.iloc[10:200])


# In[138]:


df.describe()


# In[86]:


plt.figure(figsize=(3,10))
sns.boxplot(x="source", y="retweetCount", data=df)
plt.savefig('111.png')
plt.show()


# In[7]:


plt.figure(figsize=(3,10))
sns.boxplot(x="source", y="favoriteCount", data=df)
plt.savefig('111.png')
plt.show()


# In[8]:


from statsmodels.formula.api import ols

fit = ols('retweetCount ~ C(source)', data=df).fit() 

fit.summary()


# In[ ]:





# In[9]:


def anova(var= "retweetCount ~ source"):
    lm = ols(var, data=df).fit()
    table = sm.stats.anova_lm(lm)
    print(table)


# In[10]:


anova()


# In[11]:


anova(var="favoriteCount ~ source")


#  # 1.1.3

# In[190]:


fitretweet = sm.GLM.from_formula('retweetCount ~ C(exclamation)*C(hashtag)', data=df).fit() 
fitretweet.summary()


# In[48]:


fitfavorite = sm.GLM.from_formula('favoriteCount ~ C(exclamation)*C(hashtag)', data=df).fit()
fitfavorite.summary()


# In[189]:


mewfavorite = sm.GLM.from_formula('favoriteCount ~ exclamation*hashtag + picture + C(Hourbucket)', data=df).fit()
mewfavorite.summary()


# In[191]:


mewret = sm.GLM.from_formula('retweetCount ~ exclamation*hashtag + picture + C(Hourbucket)', data=df).fit()
mewret.summary()


# In[49]:


anova(var= "retweetCount ~ exclamation*hashtag")


# In[50]:


anova(var= "favoriteCount ~ exclamation*hashtag")


# In[139]:


anova(var= "retweetCount ~ exclamation*hashtag+ picture + C(Hourbucket)")


# In[140]:


anova(var= "favoriteCount ~ exclamation*hashtag+ picture + C(Hourbucket)")


# In[52]:


anova(var= "favoriteCount ~ exclamation*hashtag+ picture + C(hour)")


# In[149]:


iphone = df[df.source == 'iPhone']


# In[150]:


iphone.describe()


# In[151]:


android = df[df.source == 'Android']


# In[152]:


android.describe()


# In[153]:


def anovaiphone(var= "retweetCount ~ exclamation*hashtag+ picture + C(Hourbucket)"):
    lm = ols(var, data=iphone).fit()
    table = sm.stats.anova_lm(lm)
    print(table)


# In[154]:


anovaiphone()


# In[158]:


anovaiphone(var= "favoriteCount ~ exclamation*hashtag + C(Hourbucket) + picture")


# In[ ]:





# In[170]:


def anovaandroid(var= "retweetCount ~ exclamation*hashtag+ picture +C(Hourbucket)"):
    lm = ols(var, data=android).fit()
    table = sm.stats.anova_lm(lm)
    print(table)


# In[171]:


anovaandroid()


# In[175]:


anovaandroid(var = "favoriteCount ~ exclamation*hashtag+ picture +C(Hourbucket)")


# In[176]:


anovaandroid(var = "retweetCount ~ exclamation+hashtag+ picture + hour")


# In[207]:

import pdb
pdb.set_trace()
#Fav Calculations
niphone = 296
nandroid = 373
varfaviphone = 44744.768687**2
varfavandroid = 62281.564310**2
sseiphone = 4.967045e+11
sseandroid = 1.367431e+12 
degree = 7
seiphone = ((sseiphone**2)/(niphone-degree-1))**0.5
seandroid = ((sseandroid**2)/(nandroid-degree-1))**0.5
Sb1 = np.sqrt(seiphone*seiphone / (niphone-1) / varfaviphone)
Sb2 = np.sqrt(seandroid*seandroid / (nandroid-1) / varfavandroid)
s = np.sqrt((Sb1*Sb1)+(Sb2*Sb2))


# In[204]:


Sb1


# In[205]:


Sb2


# In[206]:


s


# In[172]:


retweetandroid = sm.GLM.from_formula('retweetCount ~ exclamation*hashtag+ picture + C(Hourbucket)', data=android).fit()
retweetandroid.summary()


# In[174]:


retweetiphone = sm.GLM.from_formula('retweetCount ~ exclamation*hashtag+ picture + C(Hourbucket)', data=iphone).fit()
retweetiphone.summary()


# In[183]:


favandroid = sm.GLM.from_formula('favoriteCount ~ exclamation*hashtag+ picture + C(Hourbucket)', data=android).fit()
favandroid.summary()


# In[184]:


faviphone = sm.GLM.from_formula('favoriteCount ~ exclamation*hashtag+ picture + C(Hourbucket)', data=iphone).fit()
faviphone.summary()


# In[ ]:





# In[ ]:





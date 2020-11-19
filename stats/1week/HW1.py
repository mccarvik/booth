"""
HW 1 for stats
"""
import sys
import pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import statsmodels.api as stats
from pandas.plotting import scatter_matrix

mpl.use('Agg')

# Question 2
# tsy_df = pd.read_csv("treasury.csv")
# tsy_df = tsy_df.set_index('Date').pct_change().iloc[1:]
# tsy_df.to_csv("tsy_daily_chg.csv")
# sns.heatmap(tsy_df.corr(), cmap="RdBu_r", annot=True)
# plt.savefig("./corrplot.png", dpi=400)
# plt.close()
# df = pd.DataFrame(tsy_df[["1 Mo", "2 Mo", "3 Mo", "20 Yr", "30 Yr"]])
# scatter_matrix(df, alpha = 0.2, figsize = (6, 6), diagonal = 'kde')
# plt.savefig("./scatter_matrix.png", dpi=400)
# plt.close()

# Question 3
aig_df = pd.read_csv("aig.csv")
pdb.set_trace()
# aig_df['model'] = 4 - 0.005 *  (aig_df['Time'] - 93500)
# var = ((aig_df['model'] - aig_df['Price'])**2).mean()
# aig_df['model'] = 3.89 - 0.005 *  (aig_df['Time'] - 93500)
# var = ((aig_df['model'] - aig_df['Price'])**2).mean()
aig_df['shift'] = aig_df['Price'].shift(1)
aig_df['shift_resid'] = aig_df['Price'] - aig_df['shift']
print((aig_df['shift_resid']**2).mean())

print(aig_df.corr())
sns.lmplot(x='Time',y='Price',data=aig_df,fit_reg=True) 
plt.savefig("./linreg.png", dpi=400)
plt.close()
result = sm.ols(formula="Price ~ Time", data=aig_df).fit()
pdb.set_trace()
print(abs(result.resid).mean())
print(((result.resid)**2).mean())

# Question 4
rent_df = pd.read_csv("rent.csv")
# rent_df[['Rent']].boxplot()
# plt.savefig("./boxplot_marginal.png", dpi=400)
# plt.close()
# rent_df[['Rent', 'AC']].boxplot(by='AC')
# plt.savefig("./boxplot_conditional_ac.png", dpi=400)
# plt.close()
# rent_df[['Rent', 'Rooms']].boxplot(by='Rooms')
# plt.savefig("./boxplot_conditional_rooms.png", dpi=400)
# plt.close()

# how to do it manually
# model = sm.ols(formula="Rent ~ AC", data=rent_df).fit()
# sst = ((rent_df['Rent'] - rent_df['Rent'].mean())**2).sum()
# ssr = ((model.resid)**2).sum()
# sse = sst - ssr

# AC and rooms
# model = sm.ols(formula="Rent ~ AC", data=rent_df).fit()
# anova_ac = stats.stats.anova_lm(model, typ=2)
# ssr = anova_ac.loc['AC']['sum_sq']
# sse = anova_ac.loc['Residual']['sum_sq']
# sst = ssr + sse
# print("AC")
# print("sst:  {}    ssr:  {}   sse:  {}".format(sst, ssr, sse))
# print("ssr / sst:  {}".format(str(ssr / sst)))
# print(anova_ac)

# model = sm.ols(formula="Rent ~ Rooms", data=rent_df).fit()
# anova_ac = stats.stats.anova_lm(model, typ=2)
# ssr = anova_ac.loc['Rooms']['sum_sq']
# sse = anova_ac.loc['Residual']['sum_sq']
# sst = ssr + sse
# print("Rooms")
# print("sst:  {}    ssr:  {}   sse:  {}".format(sst, ssr, sse))
# print("ssr / sst:  {}".format(str(ssr / sst)))
# pdb.set_trace()
# print(anova_ac)

# SqFt
sys.exit()
rent_sqft = rent_df[['Rent', 'SqFt']]
# correlation
corr = rent_sqft.corr().values[0][1]
print(corr)
# rent stdev
print(rent_sqft['SqFt'].std())
# rent stdev
print(rent_sqft['Rent'].std())
# beta
beta = corr * (rent_sqft['Rent'].std() / rent_sqft['SqFt'].std())
print(beta)
model = sm.ols(formula="Rent ~ SqFt", data=rent_sqft).fit()

sns.lmplot(x='SqFt',y='Rent',data=rent_sqft,fit_reg=True) 
plt.savefig("./linreg_rent.png", dpi=400)
plt.close()

pdb.set_trace()
result = sm.ols(formula="Rent ~ SqFt", data=rent_sqft).fit()
rent_sqft['resid'] = result.resid
rent_sqft['resid'].hist()
plt.savefig("./rent_hist.png", dpi=400)
plt.close()
rent_sqft.plot.scatter(x='SqFt', y='resid', c='DarkBlue')
plt.savefig("./resid_sqft.png", dpi=400)
plt.close()

# removing outliers using IQR
Q1 = rent_sqft['SqFt'].quantile(0.25)
Q3 = rent_sqft['SqFt'].quantile(0.75)
IQR = Q3 - Q1
rent_no_outlier = rent_sqft[(rent_sqft['SqFt'] > (Q1 - 1.5 * IQR))]
rent_no_outlier = rent_sqft[(rent_sqft['SqFt'] < (Q3 + 1.5 * IQR))]

sns.lmplot(x='SqFt',y='Rent',data=rent_no_outlier,fit_reg=True) 
plt.savefig("./linreg_rent_no_outlier.png", dpi=400)
plt.close()

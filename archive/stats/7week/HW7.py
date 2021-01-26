"""
HW7 for Applied Regression
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

warnings.filterwarnings("ignore")



def prob1():
    ojdf = pd.read_csv("OJ.csv")
    ojdf['log_mn_vol'] = np.log(ojdf['minutevol'])
    ojdf['log_mn_px'] = np.log(ojdf['minuteprice'])
    ojdf['log_dm_px'] = np.log(ojdf['dmnckprice'])
    ojdf['log_trop_px'] = np.log(ojdf['tropicprice'])
    # pd.plotting.scatter_matrix(ojdf[["minutevol","tropicprice","minuteprice","dmnckprice"]])
    # plt.savefig("./scatter_mat_oj.png", dpi=400)
    # plt.close()
    # pd.plotting.scatter_matrix(ojdf[["log_mn_vol","log_mn_px","log_dm_px","log_trop_px"]])
    # plt.savefig("./scatter_mat_oj_logvol.png", dpi=400)
    # plt.close()
    
    # ojdf[['log_mn_vol', 'minutead']].boxplot(by='minutead')
    # plt.savefig("./boxplot_minutead.png", dpi=400)
    # plt.close()
    # ojdf[['log_mn_vol', 'tropicad']].boxplot(by='tropicad')
    # plt.savefig("./boxplot_tropad.png", dpi=400)
    # plt.close()
    # ojdf[['log_mn_vol', 'dmnckad']].boxplot(by='dmnckad')
    # plt.savefig("./boxplot_dmnckad.png", dpi=400)
    # plt.close()
    
    # forward stepwise
    # covars = ['tropicad', 'dmnckad', 'minutead', 'log_mn_px', 'log_dm_px', 'log_trop_px']
    
    # cols = []
    # formula = "log_mn_vol ~ "
    # formbic = 10000000000
    # while True:
    #     minbic = 10000000000000
    #     if len(formula) > 10:
    #         formula = formula + " + "
    #     for cov in covars:
    #         if cov in cols:
    #             continue
    #         reg = sm.ols(formula=formula + cov, data=ojdf).fit()
    #         if reg.bic < minbic:
    #             minbic = reg.bic
    #             mincov = cov
    #         # anova = stats.stats.anova_lm(reg, typ=2)
    #         # print(reg.summary())
    #         # print(anova)
    #     if minbic > formbic:
    #         break
    #     formula = formula + mincov
    #     formbic = minbic
    #     cols.append(mincov)
    #     print(formula)
    #     print("BIC:" + str(round(minbic,3)))
    #     reg = sm.ols(formula=formula, data=ojdf).fit()
    #     anova = stats.stats.anova_lm(reg, typ=2)
    #     print(reg.summary())
    #     print(anova)
    
    pdb.set_trace()
    formula = "log_mn_vol ~ minutead + log_mn_px + log_dm_px + log_trop_px + log_mn_px*minutead + log_trop_px*minutead"
    reg = sm.ols(formula=formula, data=ojdf).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    print("BIC: " + str(reg.bic))


    for var1 in ['minutead', 'log_mn_px', 'log_dm_px', 'log_trop_px']:
        for var2 in ['minutead', 'log_mn_px', 'log_dm_px', 'log_trop_px']:
            # if var1 == var2 or var1 + "*" + var2 in ["minutead*minuteprice", "minuteprice*minutead", "tropicprice*minutead", "minutead*tropicprice"]:
            if var1 == var2 or var1 + "*" + var2 in []:
                continue
            pdb.set_trace()
            reg = sm.ols(formula=formula + "+ {}".format(var1 + "*" + var2), data=ojdf).fit()
            anova = stats.stats.anova_lm(reg, typ=2)
            print(reg.summary())
            print(anova)
            print(reg.bic)


def prob2():
    okun = pd.read_csv("okunIV.csv")
    okun['ue_diff'] = okun['UNEMPLOYMENT'].diff(1)
    okun['gdp_diff'] = okun['GDP'].diff(1)
    
    # okun.plot.scatter(x="GDP", y="ue_diff")
    # plt.savefig("okun_orig_scatter.png", dpi=400)
    # plt.close()
    
    # reg = sm.ols(formula="ue_diff ~ GDP", data=okun).fit()
    # okun['resid_orig'] = reg.resid
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    
    print(okun.corr())
    
    reg = sm.ols(formula="ue_diff ~ PUBLIC_CONSUMP", data=okun).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    
    
def prob3():
    bike_orig = pd.read_csv("bikeSharing.csv")
    bike = bike_orig
    
    # remove outliers
    bike_temp = bike[['atemp', 'temp']]
    bike_temp['diff'] = bike['atemp'] - bike['temp']
    bike_temp = bike_temp[['diff']]
    zs = scipy.zscore(bike_temp)
    abs_z_scores = np.abs(zs)
    filtered_entries = (abs_z_scores < 4).all(axis=1)
    bike = bike[filtered_entries]
    
    pdb.set_trace()
    bike = bike[bike['weather'] != 4]
    
    # pd.plotting.scatter_matrix(bike[['temp', 'atemp', 'windspeed', 'humidity', 'count']])
    # plt.savefig("./scatter_mat_bike.png", dpi=400)
    # plt.savefig("./scatter_mat_bike_fixed.png", dpi=400)
    # plt.close()
    bike['datetime'] = pd.to_datetime(bike['datetime'])
    bike['w1'] = bike.apply(lambda x: 1 if x['weather'] == 1 else 0, axis=1)
    bike['w2'] = bike.apply(lambda x: 1 if x['weather'] == 2 else 0, axis=1)
    bike['w3'] = bike.apply(lambda x: 1 if x['weather'] == 3 else 0, axis=1)
    # bike['w4'] = bike.apply(lambda x: 1 if x['weather'] == 4 else 0, axis=1)
    bike['s1'] = bike.apply(lambda x: 1 if x['season'] == 1 else 0, axis=1)
    bike['s2'] = bike.apply(lambda x: 1 if x['season'] == 2 else 0, axis=1)
    bike['s3'] = bike.apply(lambda x: 1 if x['season'] == 3 else 0, axis=1)
    bike['s4'] = bike.apply(lambda x: 1 if x['season'] == 4 else 0, axis=1)
    train = bike[:int(len(bike)*0.75)]
    test = bike[int(len(bike)*0.75):]
    
    
    # forward stepwise weather
    covars = ['s1', 's2', 's3', 's4', 'holiday', 'workingday', 'w1', 'w2', 'w3', 'temp', 'atemp' , 'humidity', 'windspeed']
    
    # cols = []
    # formula = "count ~ "
    # formbic = 10000000000
    # while True:
    #     minbic = 10000000000000
    #     if len(formula) > 10:
    #         formula = formula + " + "
    #     for cov in covars:
    #         if cov in cols:
    #             continue
    #         reg = sm.ols(formula=formula + cov, data=train).fit()
    #         if reg.bic < minbic:
    #             minbic = reg.bic
    #             mincov = cov
    #         # anova = stats.stats.anova_lm(reg, typ=2)
    #         # print(reg.summary())
    #         # print(anova)
    #     if minbic > formbic:
    #         break
    #     formula = formula + mincov
    #     formbic = minbic
    #     cols.append(mincov)
    #     print(formula)
    #     print("BIC:" + str(round(minbic,3)))
    #     reg = sm.ols(formula=formula, data=train).fit()
    #     anova = stats.stats.anova_lm(reg, typ=2)
    #     print(reg.summary())
    #     print(anova)
    
    # train['winter'] = train.apply(lambda x: 1 if x['datetime'].month in [12, 1, 2, 3] else 0, axis=1)
    # formula = "count ~ temp + humidity + s3 + s4 + temp*humidity"
    # reg = sm.ols(formula=formula, data=train).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC: " + str(reg.bic))
    
    # for var1 in ['s3', 's4']:
    #     for var2 in  ['temp', 'humidity']:
    #         if var1 == var2 or var1 + "*" + var2 in []:
    #             continue
    #         pdb.set_trace()
    #         reg = sm.ols(formula="count ~ humidity + temp + s3 + s4 + {}".format(var1 + "*" + var2), data=train).fit()
    #         anova = stats.stats.anova_lm(reg, typ=2)
    #         print(reg.summary())
    #         print(anova)
    #         print(reg.bic)
    
    
    
    #### Part 3

    # forward stepwise
    covars = np.arange(24)
    
    cols = []
    formula = "count ~ temp + humidity + s3 + s4 + temp*humidity "
    train = train[train['s1'] == 1]
    print("s1 = True")
    # formula = "casual ~ temp + workingday + holiday + humidity + s3 + "
    # formula = "registered ~ temp + workingday + humidity + s3 + s4 + "
    # formbic = 10000000000
    # while True:
    #     minbic = 10000000000000
    #     if len(formula) > 10:
    #         formula = formula + " + "
    #     for cov in covars:
    #         if cov in cols:
    #             continue
    #         train['test'] = train.apply(lambda x: 1 if x['datetime'].hour == cov else 0, axis=1)
    #         reg = sm.ols(formula=formula + "test", data=train).fit()
    #         if reg.bic < minbic:
    #             minbic = reg.bic
    #             mincov = cov
    #         anova = stats.stats.anova_lm(reg, typ=2)
    #         # print(reg.summary())
    #         # print(anova)
    #         print("Hour: " + str(cov), end="   ")
    #         print("BIC:" + str(round(reg.bic,3)), end="   ")
    #         print("R^2:" + str(round(reg.rsquared,3)))
    #     return
    #     pdb.set_trace()
    #     if minbic > formbic:
    #         break
    #     formula = formula + mincov
    #     formbic = minbic
    #     cols.append(mincov)
    #     print(formula)
    #     print("BIC:" + str(round(minbic,3)))
    #     reg = sm.ols(formula=formula, data=ojdf).fit()
    #     anova = stats.stats.anova_lm(reg, typ=2)
    #     print(reg.summary())
    #     print(anova)
    
    
    # formula = "count ~ temp + humidity + temp*humidity + s3 + s4 + hr8 + hr17 + hr18"
    train['hr8'] = train.apply(lambda x: 1 if x['datetime'].hour == 8 else 0, axis=1)
    train['hr17'] = train.apply(lambda x: 1 if x['datetime'].hour == 17 else 0, axis=1)
    train['hr18'] = train.apply(lambda x: 1 if x['datetime'].hour == 18 else 0, axis=1)
    # reg = sm.ols(formula=formula, data=train).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC: " + str(reg.bic))
    
    # for var1 in ['humidity', 'temp']:
    #     for var2 in  ['hr8', 'hr17', 'hr18']:
    #         if var1 == var2 or var1 + "*" + var2 in []:
    #             continue
    #         reg = sm.ols(formula="count ~ temp + humidity + temp*humidity + s3 + s4+ hr8 + hr17 + hr18 + {}".format(var1 + "*" + var2), data=train).fit()
    #         anova = stats.stats.anova_lm(reg, typ=2)
    #         pdb.set_trace()
    #         print(reg.summary())
    #         print(anova)
    #         print(reg.bic)
    
    
    ########## Part 4
    
    
    # formula = "casual ~ temp + workingday + holiday + temp*workingday + humidity + s3 + rush + day + night + temp*night"
    # formula = "registered ~ workingday + temp + humidity + temp*workingday + s3 + s4 + rush + night + temp*rush"
    # formula = "count ~ temp + humidity + temp*humidity + s3 + s4 + rush + day + night"
    # formula = "count ~ temp + humidity + temp*humidity + s3 + s4 + rush + night + temp*rush"
    # train['rush'] = train.apply(lambda x: 1 if x['datetime'].hour in [7, 8, 16, 17, 18, 19] else 0, axis=1)
    # train['day'] = train.apply(lambda x: 1 if x['datetime'].hour in [9, 10, 11, 12, 13, 14, 15] else 0, axis=1)
    # train['night'] = train.apply(lambda x: 1 if x['datetime'].hour in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6] else 0, axis=1)
    # reg = sm.ols(formula=formula, data=train).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC: " + str(reg.bic))
    
    # for var1 in ['humidity', 'temp']:
    #     for var2 in  ['rush', 'night']:
    #         if var1 == var2 or var1 + "*" + var2 in []:
    #             continue
    #         pdb.set_trace()
    #         reg = sm.ols(formula="registered ~ workingday + temp + humidity + temp*workingday + s3 + s4 + rush + night + {}".format(var1 + "*" + var2), data=train).fit()
    #         anova = stats.stats.anova_lm(reg, typ=2)
    #         print(reg.summary())
    #         print(anova)
    #         print(reg.bic)
    
    
    # formula = "casual ~ temp + workingday + holiday + temp*workingday + humidity + s3 + hr17 + hr18 + temp*hr18"
    formula = "registered ~ workingday + temp + humidity + temp*workingday + s3 + s4 + hr8 + hr17 + hr18 + temp*hr18"
    # formula = "count ~ temp + humidity + temp*humidity + s3 + s4 + hr8 + hr17 + hr18 + temp*hr18"
    # reg = sm.ols(formula=formula, data=train).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC: " + str(reg.bic))
    test['hr8'] = test.apply(lambda x: 1 if x['datetime'].hour == 8 else 0, axis=1)
    test['hr17'] = test.apply(lambda x: 1 if x['datetime'].hour == 17 else 0, axis=1)
    test['hr18'] = test.apply(lambda x: 1 if x['datetime'].hour == 18 else 0, axis=1)
    reg = sm.ols(formula=formula, data=test).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    print("BIC: " + str(reg.bic))
    print("MSE:  " + str((reg.resid**2).mean()))
    

    # formula = "casual ~ temp + workingday + holiday + temp*workingday + humidity + s3 + rush + day + night + temp*night"
    formula = "registered ~ workingday + temp + humidity + temp*workingday + s3 + s4 + rush + night + temp*rush"
    # formula = "count ~ temp + humidity + temp*humidity + s3 + s4 + rush + night + temp*rush"
    test['rush'] = test.apply(lambda x: 1 if x['datetime'].hour in [7, 8, 16, 17, 18, 19] else 0, axis=1)
    test['day'] = test.apply(lambda x: 1 if x['datetime'].hour in [9, 10, 11, 12, 13, 14, 15] else 0, axis=1)
    test['night'] = test.apply(lambda x: 1 if x['datetime'].hour in [20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6] else 0, axis=1)
    reg = sm.ols(formula=formula, data=test).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    pdb.set_trace()
    print(reg.summary())
    print(anova)
    print("BIC: " + str(reg.bic))
    print("MSE:  " + str((reg.resid**2).mean()))
    
    

# prob1()
# prob2()
prob3()
"""
Final for Applied Regression
"""

import pdb
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians
from sklearn import linear_model
import statsmodels.formula.api as sm
import statsmodels.api as stats
import scipy.stats as scipy
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import warnings

warnings.filterwarnings("ignore")


# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return diff
 
# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob


def prob1():
    # TODO
    # - only mention alpha in passing
    mkt = pd.read_csv("market.csv")
    mkt['goog_exc'] = mkt['GOOG'] - mkt['RF']
    mkt['sbux_exc'] = mkt['SBUX'] - mkt['RF']
    
    for sec in ['sbux_exc', 'goog_exc']:
        pdb.set_trace()
        formula = sec + " ~ Mkt_Minus_Rf"
        reg = sm.ols(formula=formula, data=mkt).fit()
        anova = stats.stats.anova_lm(reg, typ=2)
        print(reg.summary())
        print(anova)
        print("BIC:" + str(round(reg.bic,3)), end="   ")
        print("R^2:" + str(round(reg.rsquared,3)), end="   ")
        print("MSE:  " + str((reg.resid**2).mean()))
        
        pdb.set_trace()
        formula = sec + " ~ Mkt_Minus_Rf + SMB + HML"
        reg = sm.ols(formula=formula, data=mkt).fit()
        anova = stats.stats.anova_lm(reg, typ=2)
        print(reg.summary())
        print(anova)
        print("BIC:" + str(round(reg.bic,3)), end="   ")
        print("R^2:" + str(round(reg.rsquared,3)), end="   ")
        print("MSE:  " + str((reg.resid**2).mean()))
        
        pdb.set_trace()
        formula = sec + " ~ Mkt_Minus_Rf + SMB + HML + Mom"
        reg = sm.ols(formula=formula, data=mkt).fit()
        anova = stats.stats.anova_lm(reg, typ=2)
        print(reg.summary())
        print(anova)
        print("BIC:" + str(round(reg.bic,3)), end="   ")
        print("R^2:" + str(round(reg.rsquared,3)), end="   ")
        print("MSE:  " + str((reg.resid**2).mean()))
        
        
    pdb.set_trace()
    print()
    

def prob3():
    # TODO
    # - test dataset
    smsa = pd.read_csv("smsa.csv")
    # pd.plotting.scatter_matrix(smsa[['JanT','JulyT','RelHum','Rain','Mortality',]])
    # plt.savefig("./scatter_smsa1.png", dpi=400)
    # plt.close()
    # pd.plotting.scatter_matrix(smsa[['NonWht','WC','Pop','PHouse','Mortality']])
    # plt.savefig("./scatter_smsa2.png", dpi=400)
    # plt.close()
    # pd.plotting.scatter_matrix(smsa[['Income','HCPot','NOxPot','Mortality']])
    # plt.savefig("./scatter_smsa3.png", dpi=400)
    # plt.close()
    # pd.plotting.scatter_matrix(smsa[['Edu','PopD','S02Pot','Mortality',]])
    # plt.savefig("./scatter_smsa4.png", dpi=400)
    # plt.close()
    
    smsa['log_nx'] = np.log(smsa['NOxPot'])
    smsa['log_hc'] = np.log(smsa['HCPot'])
    smsa['log_pop'] = np.log(smsa['Pop'])
    smsa['log_so2'] = np.log(smsa['S02Pot'])
    # pd.plotting.scatter_matrix(smsa[['log_so2', 'log_pop','log_hc','log_nx','Mortality',]])
    # plt.savefig("./scatter_smsa_log.png", dpi=400)
    # plt.close()
    
    train = smsa[:int(len(smsa)*0.75)]
    test = smsa[int(len(smsa)*0.75):]
    old_cols = set(['NOxPot', 'HCPot', 'Pop', 'S02Pot', "City", "Mortality"])
    covars = list(set(train.columns) - old_cols)
    

    ##### Part 2 #####
    
    # All vars
    # formula = "Mortality ~ " + "+".join([col for col in covars])
    # reg = sm.ols(formula=formula, data=train).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    # pdb.set_trace()

    ##### Part 3 #####
    
    # formula = "Mortality ~ NonWht + PopD + Edu + log_so2 + log_nx + log_hc + JanT"
    # formula = "Mortality ~ NonWht + PopD + Edu + log_so2 + log_nx + JanT"    
    # reg = sm.ols(formula=formula, data=train).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    # train['resid'] = reg. resid
    # train.plot.scatter(x="Mortality", y="resid")
    # plt.savefig("./scatter_resid_cust.png", dpi=400)
    # plt.close()
    # pdb.set_trace()
    
    
    ##### Part 4 #####

    # Forward stepwise
    # cols = []
    # # pdb.set_trace()
    # formula = "Mortality ~ "
    # formbic = 10000000000
    # while True:
    #     minbic = 10000000000000
    #     for cov in covars:
    #         if cov in cols:
    #             continue
    #         reg = sm.ols(formula=formula + " + " + cov, data=train).fit()
    #         if reg.bic < minbic:
    #             minbic = reg.bic
    #             mincov = cov
    #         # anova = stats.stats.anova_lm(reg, typ=2)
    #         # print(reg.summary())
    #         # print(anova)
    #     if minbic > formbic:
    #         break
    #     formula = formula + " + " + mincov
    #     formbic = minbic
    #     cols.append(mincov)
    #     print(formula)
    #     print("BIC:" + str(round(minbic,3)))
    # reg = sm.ols(formula=formula, data=train).fit()
    # # pdb.set_trace()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # # print(reg.summary())
    # # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    # pdb.set_trace()
    
    # backward stepwise
    # cols = []
    # formula = "Mortality ~ "
    # for covar in covars:
    #     formula = formula + " + " + covar
    # formula = formula[:-2]
    
    # formbic = 10000000000
    # while True:
    #     minbic = 10000000000000
    #     for cov in covars:
    #         formula = "Mortality ~ "

    #         temp_covs = list(set(covars) - set([cov]))
    #         for covar in temp_covs:
    #             formula = formula + " + " + covar
    #         # formula = formula[:-2]
            
    #         formula = "Mortality ~ " + formula[12:]
    #         reg = sm.ols(formula=formula, data=train).fit()
    #         # print(formula)
    #         if reg.bic < minbic:
    #             minbic = reg.bic
    #             mincov = cov
    #     if minbic > formbic:
    #         break
    #     formbic = minbic
    #     covars = list(set(covars) - set([mincov]))
    #     print("Removed: " + mincov + "     BIC: " +  str(round(minbic,3)))
    
    # formula = "Mortality ~ "
    # for covar in covars:
    #     formula = formula + " + " + covar
    # formula = "Mortality ~ " + formula[14:]
    # print(formula)
    # # pdb.set_trace()
    # reg = sm.ols(formula=formula, data=train).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    # pdb.set_trace()
    
    # train['resid'] = reg. resid
    # train.plot.scatter(x="Mortality", y="resid")
    # plt.savefig("./scatter_resid.png", dpi=400)
    # plt.close()
    # pdb.set_trace()


    ###### Part 5 ##### 
    # # formula = "Mortality ~ NonWht + PopD + Edu + JanT + Rain + log_nx + log_so2"
    # # formula = "Mortality ~ NonWht + PopD + Edu + JanT + Rain + log_nx + log_so2 + NonWht*log_so2"
    # formula = "Mortality ~ NonWht + PopD + Edu + JanT + Rain + log_nx + log_so2 + NonWht*log_nx"
    # # formula = "Mortality ~ NonWht + PopD + Edu + JanT + Rain + log_nx + log_so2 + PopD*JanT"
    # for var1 in ['NonWht', 'PopD', 'Edu', 'JanT', 'Rain', 'log_nx', 'log_so2']:
    #     for var2 in ['NonWht', 'PopD', 'Edu', 'JanT', 'Rain', 'log_nx', 'log_so2']:
    #         # if var1 == var2 or var1 + "*" + var2 in ["minutead*minuteprice", "minuteprice*minutead", "tropicprice*minutead", "minutead*tropicprice"]:
    #         if var1 == var2 or var1 + var2 in ["NonWhtlog_so2", "log_so2NonWht"]:
    #             continue
    #         reg = sm.ols(formula=formula + "+ {}".format(var1 + "*" + var2), data=train).fit()
    #         anova = stats.stats.anova_lm(reg, typ=2)
    #         print(reg.summary())
    #         print(anova)
    #         print("BIC:" + str(round(reg.bic,3)), end="   ")
    #         print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    #         print("MSE:  " + str((reg.resid**2).mean()))
    
    # reg = sm.ols(formula=formula, data=train).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    
    # train['resid'] = reg. resid
    # train.plot.scatter(x="Mortality", y="resid")
    # plt.savefig("./scatter_resid_inter.png", dpi=400)
    # plt.close()
    # pdb.set_trace()
    
    
    ##### Part 6 #####
    
    # all vars
    # formula = "Mortality ~ " + " + ".join([col for col in covars])
    # custom
    # formula = "Mortality ~ NonWht + PopD + Edu + log_so2 + log_nx + JanT"
    # forward stepwise
    # formula = "Mortality ~ NonWht + PopD + Edu + JanT"
    # backward stepwise
    # formula = "Mortality ~ NonWht + PopD + Edu + log_so2 + log_nx + JanT + Rain"
    # stewpwise with interaction
    formula = "Mortality ~ NonWht + PopD + Edu + log_so2 + log_nx + JanT + Rain + NonWht*log_nx"
    
    reg = sm.ols(formula=formula, data=train).fit()
    # sse = ((reg.predict(test) - test['Mortality'])**2).sum()
    mse = ((reg.predict(test) - test['Mortality'])**2).mean()
    # sst = ((test['Mortality'] - test['Mortality'].mean())**2).sum()
    # ssr = ((reg.predict(test) - test['Mortality'].mean())**2).sum()
    # ssr = sst - sse
    # r2 = ssr / sst
    print(formula)
    # print("sst: " + str(round(sst,3)) + "   ssr: " + str(round(ssr,3)) + "    sse: " + str(round(sse,3)))
    # print("R^2:" + str(round(r2,3)), end="   ")
    print("MSE:  " + str(round(mse,3)))
    return


def prob2():
    # TODO:
    # - add diff term as potential seasonality
    seat = pd.read_csv("seatbelt.csv")
    # pd.plotting.scatter_matrix(seat[['drivers', 'month', 'year', 'kms', 'petrol']])
    # plt.savefig("./scatter_seat.png", dpi=400)
    # plt.close()
    
    seat['date'] = seat.apply(lambda x: dt.datetime(1900+ int(x['year']), int(x['month']), 1, 0, 0), axis=1)
    seat['log_driv'] = np.log(seat['drivers'])
    seat['season_m_sin'] = np.sin((2 * np.pi * seat['month']) / 12)
    seat['season_m_cos'] = np.cos((2 * np.pi * seat['month']) / 12)
    seat['lag_1'] = seat['log_driv'].shift(1)
    seat['lag_2'] = seat['log_driv'].shift(2)
    seat['lag_3'] = seat['log_driv'].shift(3)
    seat['lag_4'] = seat['log_driv'].shift(4)
    seat['lag_12'] = seat['log_driv'].shift(12)
    seat['lag_24'] = seat['log_driv'].shift(24)
    seat['lag_36'] = seat['log_driv'].shift(36)
    seat['lag_48'] = seat['log_driv'].shift(48)
    seat['nov'] = [1 if x == 11 else 0 for x in seat['month']]
    seat['dec'] = [1 if x == 12 else 0 for x in seat['month']]
    seat['bad_weath'] = [1 if x > 10 or x < 1 else 0 for x in seat['month']]
    diff = difference(seat['drivers'], 12)
    inverted = [inverse_difference(seat['drivers'][i], diff[i]) for i in range(len(diff))]
    seat["log_driv_diff"] = [np.NaN] * 12 + inverted
    seat['diff'] = seat['drivers'] - seat['drivers'].shift(12)
    seat['lagd_1'] = seat['diff'].shift(1)
    seat['lagd_2'] = seat['diff'].shift(2)
    seat['lagd_3'] = seat['diff'].shift(3)
    seat['lagd_4'] = seat['diff'].shift(4)
    seat['lagd_12'] = seat['diff'].shift(12)
    seat2 = seat
    seat2 = seat.dropna()
    
    # seat_group = seat[['log_driv', 'year']].groupby('year').sum()
    # seat_group.reset_index().plot(x='year', y='log_driv')
    # plt.savefig("./seat_year.png", dpi=400)
    # plt.close()

    # pd.plotting.scatter_matrix(seat[['log_driv', 'month', 'year', 'kms', 'petrol']])
    # plt.savefig("./scatter_seat.png", dpi=400)
    # plt.close()
    # seat[['log_driv', 'law']].boxplot(by='law')
    # plt.savefig("./boxplot_law.png", dpi=400)
    # plt.close()
    
    # Seasonality
    
    # formula = "log_driv ~ season_m_sin + season_m_cos"
    # # formula = "log_driv ~ bad_weath"
    # reg = sm.ols(formula=formula, data=seat).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    # pdb.set_trace()

    
    ##### original model #####
    
    # pdb.set_trace()
    # formula = "diff ~ kms + petrol + year"
    # formula = "diff ~ petrol"
    # formula = "log_driv ~ kms + petrol + year"
    # formula = "log_driv ~ petrol + year"
    # formula = "log_driv ~ petrol + year + petrol*year"
    # reg = sm.ols(formula=formula, data=seat).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    # pdb.set_trace()
    
    
    # ACF for seasonality
    
    # formula = "log_driv ~ petrol + year"
    # reg = sm.ols(formula=formula, data=seat).fit()
    # pdb.set_trace()
    # plot_acf(seat['log_driv'].dropna(), lags=60)
    # plt.savefig("./acf_orig.png", dpi=400)
    # plot_acf(seat['diff'].dropna(), lags=60)
    # plt.savefig("./acf_diff.png", dpi=400)
    # plt.close()
    # pdb.set_trace()
    
    
    ##### Add lags #####
    # formula = "diff ~ petrol + lagd_1 + lagd_2 + lagd_12"
    # formula = "diff ~ lagd_1 + lagd_2 + lagd_12"
    # formula = "log_driv ~ petrol + lag_1 + lag_12"
    # formula = "log_driv ~ petrol + lag_1 + lag_2 + lag_3 + lag_12"
    # reg = sm.ols(formula=formula, data=seat).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    # pdb.set_trace()
    
    
    # R^2 calc for diff
    
    # pdb.set_trace()
    # formula = "diff ~ lagd_1 + lagd_12"
    # reg = sm.ols(formula=formula, data=seat).fit()
    # pred = reg.predict(seat)
    # resid = seat['drivers'].shift(12) + pred
    # sse = ((resid - seat['drivers'])**2).sum()
    # sst = ((seat['drivers'] - seat['drivers'].mean())**2).sum()
    # ssr = ((reg.predict(test) - test['Mortality'].mean())**2).sum()
    # ssr = sst - sse
    # r2 = ssr / sst
    # print("sst: " + str(round(sst,3)) + "   ssr: " + str(round(ssr,3)) + "    sse: " + str(round(sse,3)))
    # print("R^2:" + str(round(r2,3)))
    # print("MSE:  " + str(round(mse,3)))
    # pdb.set_trace()
    
    
    
    
    ##### Seasonality #####
    ### sin / cos ###
    ### bucket ###
    ### diff ###
    
    formula = "log_driv ~ petrol + lag_1 + lag_12 + bad_weath"
    # formula = "log_driv ~ petrol + lag_1 + lag_12 + season_m_sin"
    # formula = "log_driv ~ petrol + lag_1 + lag_12 + season_m_sin + season_m_cos"
    # # formula = "log_driv ~ petrol + lag_1 + lag_12 + lag_24 + dec + nov"
    # # formula = "log_driv_diff ~ petrol + lag_1 + lag_12 + lag_24 "
    reg = sm.ols(formula=formula, data=seat).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # print("BIC:" + str(round(reg.bic,3)), end="   ")
    # print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    # print("MSE:  " + str((reg.resid**2).mean()))
    # seat['resid'] = reg.resid
    # seat.plot.scatter(x='log_driv', y='resid')
    # plt.savefig("./driv_resid.png", dpi=400)
    # plt.close()
    # seat.plot.scatter(x='year', y='resid')
    # plt.savefig("./year_resid.png", dpi=400)
    # plt.close()
    # pdb.set_trace()
    # seat.plot.scatter(x='date', y='log_driv')
    # plt.plot(seat['date'], reg.predict(seat), "r")
    # plt.savefig("./y_vs_real.png", dpi=400)
    plot_acf(reg.resid.dropna(), lags=60)
    plt.savefig("./acf_model.png", dpi=400)
    plt.close()
    
    
    ##### Test Law #####
    
    formula = "log_driv ~ petrol + lag_1 + lag_12 + bad_weath + law"
    # formula = "log_driv ~ petrol + lag_1 + lag_12 + bad_weath"
    # formula = "log_driv ~ petrol + lag_12 + lag_24 + lag_36 + bad_weath + law"
    # formula = "log_driv ~ petrol + lag_12 + lag_24 + bad_weath + law"
    reg = sm.ols(formula=formula, data=seat).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    print("BIC:" + str(round(reg.bic,3)), end="   ")
    print("R^2:" + str(round(reg.rsquared,3)), end="   ")
    print("MSE:  " + str((reg.resid**2).mean()))
    seat['resid'] = reg.resid
    # seat.plot.scatter(x='log_driv', y='resid')
    # plt.savefig("./driv_resid_law.png", dpi=400)
    # plt.close()
    # seat.plot.scatter(x='year', y='resid')
    # plt.savefig("./year_resid_law.png", dpi=400)
    # plt.close()
    # seat.plot.scatter(x='date', y='log_driv')
    # plt.plot(seat['date'], reg.predict(seat), "r")
    # plt.savefig("./y_vs_real_law.png", dpi=400)
    # # plt.savefig("./y_vs_real.png", dpi=400)
    # plt.close()
    plot_acf(reg.resid.dropna(), lags=60)
    plt.savefig("./acf_model_law.png", dpi=400)
    plt.close()
    
    
    
    pdb.set_trace()
    print()



# prob1()
prob2()
# prob3()


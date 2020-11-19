"""
HW6 for Applied Regression
"""

import pdb
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as sm
import statsmodels.api as stats
import scipy.stats as scipy



def prob1():
    beef = pd.read_csv("beef.csv")
    beef['log_val'] = np.log(beef['VAL'])
    beef.plot.scatter(x='VAL', y='SIZE')
    # plt.savefig("./beef_valsize_scatter.png", dpi=400)
    # plt.close()
    
    # beef.plot.scatter(x='log_val', y='YES')
    # plt.savefig("./beef_size_scatter.png", dpi=400)
    # plt.close()
    
    # beef.plot.scatter(x='log_val', y='YES')
    # plt.savefig("./beef_logval_scatter.png", dpi=400)
    # plt.close()
    
    # beef.plot.scatter(x='VAL', y='YES')
    # plt.savefig("./beef_val_scatter.png", dpi=400)
    # plt.close()

    reg = sm.ols(formula="YES ~ SIZE + np.log(VAL)", data=beef).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    
    reg = sm.ols(formula="YES ~ SIZE + log_val + SIZE*log_val", data=beef).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    

def prob2():
    wine = pd.read_csv("winequality.csv")
    
    # remove outliers
    wine_temp = wine.drop(columns = ['color', 'quality'])
    zs = scipy.zscore(wine_temp)
    abs_z_scores = np.abs(zs)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    wine = wine[filtered_entries]
    
    wine.plot.scatter(x='quality', y='volatile_acidity')
    plt.savefig("./wine_volacid.png", dpi=400)
    plt.close()
    
    wine.plot.scatter(x='quality', y='alcohol')
    plt.savefig("./wine_alcohol.png", dpi=400)
    plt.close()
    
    return
    # all covariates
    reg = sm.ols(formula="quality ~ fixed_acidity + volatile_acidity + citric_acid + residual_sugar + chlorides + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol + color", data=wine).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    # wine['resid'] = reg.resid
    # wine.plot.scatter(x='quality', y='resid')
    # plt.savefig("./wine_y_resid.png", dpi=400)
    # plt.close()
    # wine.plot.scatter(x='alcohol', y='resid')
    # plt.savefig("./wine_alc_resid.png", dpi=400)
    # plt.close()
    
    # reg = sm.ols(formula="quality ~ fixed_acidity + volatile_acidity + residual_sugar + free_sulfur_dioxide + total_sulfur_dioxide + density + pH + sulphates + alcohol + color", data=wine).fit()
    # reg = sm.ols(formula="quality ~ fixed_acidity + volatile_acidity + residual_sugar + free_sulfur_dioxide + density + pH + sulphates + alcohol + color", data=wine).fit()
    # reg = sm.ols(formula="quality ~ fixed_acidity + volatile_acidity + residual_sugar + density + pH + sulphates + alcohol + color", data=wine).fit()
    # reg = sm.ols(formula="quality ~ volatile_acidity + residual_sugar + density + pH + sulphates + alcohol + color", data=wine).fit()
    
    # reg = sm.ols(formula="quality ~ volatile_acidity + residual_sugar + density + sulphates + alcohol + color", data=wine).fit()
    # reg = sm.ols(formula="quality ~ volatile_acidity + residual_sugar + sulphates + alcohol + color", data=wine).fit()
    
    # new model with no interactions
    pdb.set_trace()
    reg = sm.ols(formula="quality ~ volatile_acidity + residual_sugar + sulphates + alcohol", data=wine).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    # wine['resid'] = reg.resid
    # wine.plot.scatter(x='quality', y='resid')
    # plt.savefig("./wine_y_simple_resid.png", dpi=400)
    # plt.close()
    # wine.plot.scatter(x='residual_sugar', y='resid')
    # plt.savefig("./wine_sugar_resid.png", dpi=400)
    # plt.close()
    
    # With interactions
    reg = sm.ols(formula="quality ~ volatile_acidity + residual_sugar + sulphates + alcohol + residual_sugar*volatile_acidity", data=wine).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    wine['resid'] = reg.resid
    wine['resid_sugar_vol_acid'] = wine['residual_sugar'] * wine['volatile_acidity']
    # wine.plot.scatter(x='quality', y='resid')
    # plt.savefig("./wine_y_inter_resid.png", dpi=400)
    # plt.close()
    # wine.plot.scatter(x='resid_sugar_vol_acid', y='resid')
    # plt.savefig("./wine_inter_resid.png", dpi=400)
    # plt.close()
    
    BICS = [14246.71, 14336.06, 14303.64]
    for BIC in BICS:
        print(np.exp(-.5*(BIC-min(BICS))))
        print(np.exp(-.5*(BIC-min(BICS))) / sum(BICS))

    return
    # forward stepwise
    for var1 in ['volatile_acidity', 'residual_sugar', 'sulphates', 'alcohol']:
        for var2 in ['volatile_acidity', 'residual_sugar', 'sulphates', 'alcohol']:
            if var1 == var2 or var1 + "*" + var2 in ['residual_sugar*volatile_acidity', 'volatile_acidity*residual_sugar', 'residual_sugar*sulphates', 'sulphates*residual_sugar']:
                continue
                
            reg = sm.ols(formula="quality ~ volatile_acidity + residual_sugar + sulphates + alcohol + residual_sugar*volatile_acidity + residual_sugar*sulphates + {}".format(var1 + "*" + var2), data=wine).fit()
            anova = stats.stats.anova_lm(reg, typ=2)
            print(reg.summary())
            print(anova)
    

def prob3():
    nut = pd.read_csv("nutrition.csv")
    
    # nut.plot.scatter(x='age', y='woh')
    # plt.savefig("./nutrition_scatter.png", dpi=400)
    # plt.close()
    
    # reg = sm.ols(formula="woh ~ age", data=nut).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # nut['resid'] = reg.resid
    # nut.plot.scatter(x='age', y='resid')
    # plt.savefig("./nutrition_scatter_resid.png", dpi=400)
    # plt.close()
    
    # reg = sm.ols(formula="woh ~ np.log(age)", data=nut).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # nut['resid'] = reg.resid
    # nut.plot.scatter(x='age', y='resid')
    # plt.savefig("./nutrition_scatter_logresid.png", dpi=400)
    # plt.close()
    
    nut['age7'] = nut['age'] > 7
    # reg = sm.ols(formula="woh ~ age + age7 + age*age7", data=nut).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)
    # nut['resid'] = reg.resid
    # nut.plot.scatter(x='age', y='resid')
    # plt.savefig("./nutrition_scatter_age7resid.png", dpi=400)
    # plt.close()
    
    nut.plot.scatter(x='age', y='woh')
    x_vals = np.linspace(0,80,100)
    x_vals_age7 = x_vals > 7
    x_vals_age7 = np.array([1 if x else 0 for x in x_vals_age7])
    reg = sm.ols(formula="woh ~ age", data=nut).fit()
    y_vals = reg.params[0] + reg.params[1] * x_vals
    plt.plot(x_vals, y_vals, '--b', label="age")
    reg = sm.ols(formula="woh ~ np.log(age)", data=nut).fit()
    y_vals = reg.params[0] + reg.params[1] * np.log(x_vals)
    plt.plot(x_vals, y_vals, '--g', label="log(age)")
    reg = sm.ols(formula="woh ~ age + age7 + age*age7", data=nut).fit()
    pdb.set_trace()
    y_vals = reg.params[0] + reg.params[2] * x_vals + reg.params[1] * x_vals_age7 + reg.params[3] * (x_vals_age7 * x_vals)
    plt.plot(x_vals, y_vals, '--r', label="age + age7 + age*age7")
    plt.legend()
    plt.savefig("./nutrition_regression.png", dpi=400)
    plt.close()


def prob4():
    # burr = pd.read_csv("burrito.csv")
    burr = pd.read_csv("clairemont.csv")
    covars = list(set(burr.columns) - set(["overall"]))
    
    cols = []
    formula = "overall ~ "
    formbic = 10000000000
    while True:
        minbic = 10000000000000
        if len(formula) > 10:
            formula = formula + " + "
        for cov in covars:
            if cov in cols:
                continue
            reg = sm.ols(formula=formula + cov, data=burr).fit()
            if reg.bic < minbic:
                minbic = reg.bic
                mincov = cov
            # anova = stats.stats.anova_lm(reg, typ=2)
            # print(reg.summary())
            # print(anova)
        if minbic > formbic:
            break
        formula = formula + mincov
        formbic = minbic
        cols.append(mincov)
        print(formula)
        print("BIC:" + str(round(minbic,3)))
    
    pdb.set_trace()
    reg = sm.ols(formula=formula[:-2], data=burr).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    print(reg.summary())
    print(anova)
    
    # cols = []
    # formula = "overall ~ "
    # for covar in covars:
    #     formula = formula + " + " + covar
    # formula = formula[:-2]
    
    # formbic = 10000000000
    # while True:
    #     minbic = 10000000000000
    #     for cov in covars:
    #         formula = "overall ~ "

    #         temp_covs = list(set(covars) - set([cov]))
    #         for covar in temp_covs:
    #             formula = formula + " + " + covar
    #         # formula = formula[:-2]
            
    #         formula = "overall ~ " + formula[12:]
    #         reg = sm.ols(formula=formula, data=burr).fit()
    #         # print(formula)
    #         if reg.bic < minbic:
    #             minbic = reg.bic
    #             mincov = cov
    #     if minbic > formbic:
    #         break
    #     formula = formula + mincov
    #     formbic = minbic
    #     covars = list(set(covars) - set([mincov]))
    #     print("Removed: " + mincov + "     BIC: " +  str(round(minbic,3)))
    
    # formula = "overall ~ "
    # temp_covs = list(set(covars) - set([cov]))
    # for covar in temp_covs:
    #     formula = formula + " + " + covar
    # formula = "overall ~ " + formula[12:]
    # print(formula)    
    # reg = sm.ols(formula=formula, data=burr).fit()
    # anova = stats.stats.anova_lm(reg, typ=2)
    # print(reg.summary())
    # print(anova)

    

# prob1()
prob2()
# prob3()
# prob4()
"""
HW4 for Applied Regression
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
import pylab 
from sklearn.preprocessing import PolynomialFeatures
import skfda
import skfda.preprocessing.smoothing.kernel_smoothers as ks
import skfda.preprocessing.smoothing.validation as val


def prob1():
    opts = pd.read_csv("options.csv")
    # opts.plot.scatter(x="Strike", y="Price")
    # plt.savefig("strike_price.png", dpi=400)
    # plt.close()

    print()
    # reg = sm.ols(formula="Price ~ Strike", data=opts).fit()
    # reg = sm.ols(formula="Price ~ I(1/Strike)", data=opts).fit()
    reg = sm.ols(formula="Price ~ Strike + np.log(Strike)", data=opts).fit()
    beta = reg.params[1]
    beta_sqr = reg.params[2]
    alpha = reg.params['Intercept']
    x_vals = np.linspace(200,1800,1000)
    y_vals = alpha + beta * x_vals + beta_sqr * np.log(x_vals)
    # y_vals = alpha + beta * 1 / x_vals
    opts.plot.scatter(x="Strike", y="Price")
    plt.plot(x_vals, y_vals, '-b', label='Best Fit')
    # plt.savefig("strike_price_fit.png", dpi=400)
    # plt.savefig("strike_price_fit2.png", dpi=400)
    plt.close()
    
    opts.plot.scatter(x="Strike", y="Price")
    x_vals = np.linspace(200,1800,1000)
    y_vals = [bs_calc(x) for x in x_vals]
    plt.plot(x_vals, y_vals, '-b')
    plt.title("Vol = 0.40")
    plt.savefig("strike_price_bs.png", dpi=400)
    plt.close()
    

def bs_calc(K, S=683.22, sigma=0.40, r=0.005, T=147/365):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    call = (S * scipy.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * scipy.norm.cdf(d2, 0.0, 1.0))
    call = call * np.e ** (-r * (T))
    return call



def prob2():
    che = pd.read_csv("cheese.csv")
    che['log_vol'] = np.log(che['vol'])
    che['log_price'] = np.log(che['price'])
    # che[['log_vol', 'disp']].boxplot(by='disp')
    # plt.savefig("./boxplot_disp.png", dpi=400)
    # plt.close()
    
    model = sm.ols(formula= "log_vol ~ disp", data=che).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    ssr = anova.loc['disp']['sum_sq']
    sse = anova.loc['Residual']['sum_sq']
    sst = ssr + sse
    print("Winter")
    print("sst:  {}    ssr:  {}   sse:  {}".format(round(sst,2), round(ssr,2), round(sse,2)))
    print("ssr / sst (R^2):  {}".format(str(round(ssr / sst,2))))
    print(anova)

    # without instore displays
    pdb.set_trace()
    che_ndisp = che[che['disp'] == 0]
    che_ndisp.plot.scatter(x='log_price', y='log_vol')
    reg = sm.ols(formula="log_vol ~ log_price", data=che_ndisp).fit()
    beta = reg.params[1]
    print(beta)
    alpha = reg.params['Intercept']
    x_vals = np.linspace(0.25,1.75,100)
    y_vals = alpha + beta * x_vals
    pdb.set_trace()
    plt.plot(x_vals, y_vals, '-b')
    plt.title("Disp = 0")
    plt.savefig("./log_log_without_disp.png", dpi=400)
    plt.close()
    
    # without instore displays
    che_ydisp = che[che['disp'] == 1]
    che_ydisp.plot.scatter(x='log_price', y='log_vol')
    reg = sm.ols(formula="log_vol ~ log_price", data=che_ydisp).fit()
    beta = reg.params[1]
    print(beta)
    pdb.set_trace()
    alpha = reg.params['Intercept']
    x_vals = np.linspace(0.25,1.75,100)
    y_vals = alpha + beta * x_vals
    plt.plot(x_vals, y_vals, '-b')
    plt.title("Disp = 1")
    plt.savefig("./log_log_with_disp.png", dpi=400)
    plt.close()


def prob4():
    flu = pd.read_csv("flu.csv")
    flu['winter'] = flu.apply(lambda x: 0 if x['Week'] > 8 and x['Week'] < 48 else 1, axis=1)
    flu.rename(columns = {'Percent of Deaths Due to Pneumonia and Influenza': 'perc_death'}, inplace=True)
    flu[['perc_death', 'winter']].boxplot(by='winter')
    plt.savefig("./boxplot_conditional_winter.png", dpi=400)
    plt.close()
    
    # Winter
    # model = sm.ols(formula= "perc_death ~ winter", data=flu).fit()
    # anova = stats.stats.anova_lm(model, typ=2)
    # ssr = anova.loc['winter']['sum_sq']
    # sse = anova.loc['Residual']['sum_sq']
    # sst = ssr + sse
    # print("Winter")
    # print("sst:  {}    ssr:  {}   sse:  {}".format(round(sst,2), round(ssr,2), round(sse,2)))
    # print("ssr / sst (R^2):  {}".format(str(round(ssr / sst,2))))
    # print(anova)
    # flu.plot.scatter(x='Week', y='perc_death')
    # plt.savefig("./scatter_winter.png", dpi=400)
    # plt.close()
    
    # flu.plot.scatter(x='Week', y='perc_death')
    # reg = sm.ols(formula="perc_death ~ Week", data=flu).fit()
    # beta = reg.params[1]
    # alpha = reg.params['Intercept']
    # x_vals = np.linspace(0,52,100)
    # y_vals = alpha + beta * x_vals
    # plt.plot(x_vals, y_vals, '-b')
    # plt.savefig("./scatter_winter_regr.png", dpi=400)
    # plt.close()

    # flu['resid'] = reg.resid.values
    # flu.plot.scatter(x="Week", y='resid')
    # plt.ylabel("Residual", fontsize=12)
    # plt.savefig("resid_scatter.png", dpi=400)
    # plt.close()
    
    # scipy.probplot(y_vals, dist="norm", plot=pylab)
    # pylab.show()
    # plt.savefig("qq_plot.png", dpi=400)
    # plt.close()
    polynomial_features= PolynomialFeatures(degree=2)
    xp = polynomial_features.fit_transform(flu['Week'].values.reshape(-1,1))
    model = sm.OLS(flu['perc_death'], xp).fit()
    x_vals = np.linspace(0,52,100)
    y_vals = model.params[0] + model.params[1] * x_vals + model.params[2] * x_vals**2
    # ypred = model.predict(xp)
    # flu.plot.scatter(x='Week', y='perc_death')
    # plt.plot(x_vals, y_vals, '-b')
    
    # beta = reg.params[1]
    # alpha = reg.params['Intercept']
    # x_vals = np.linspace(0,52,100)
    # y_vals = alpha + beta * x_vals
    # plt.plot(x_vals, y_vals, '-b')
    # plt.savefig("./scatter_winter_regr2.png", dpi=400)
    # plt.close()
    
    
    xs = flu['Week']
    ys = flu['perc_death']
    pdb.set_trace()
    param_values = np.linspace(start=2, stop=25, num=24)
    knn = val.SmoothingParameterSearch(ks.KNeighborsSmoother(), param_values)
    knn.fit(fd)
    knn_fd = knn.transform(fd)
    
    
    k1 = KernelReg(xs.values, ys.values, var_type='c')
    x_vals = np.linspace(0,52,100)
    y_vals, _ = k1.fit(x_vals)
    flu.plot.scatter(x='Week', y='perc_death')
    plt.plot(x_vals, y_vals, '-b', linewidth=2)
    plt.savefig("./ksmooth.png", dpi=400)
    plt.close()


# prob1()
prob2()
# prob4()

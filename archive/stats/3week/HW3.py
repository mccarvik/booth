"""
HW3 for Applied Regression
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
import scipy.stats as scipy_int


def prob1():
    conf_int = scipy_int.norm.interval(0.9, loc=0, scale=0.02711201)
    print(conf_int)
    axes = plt.gca()
    axes.set_xlim([-0.10, 0.10])
    axes.set_ylim([-0.25, 0.25])
    
    x_vals = np.array(axes.get_xlim())
    y_vals = -0.008194 + 1.690067 * x_vals
    y_vals1 = conf_int[0] -0.008194 + 1.690067 * x_vals
    y_vals2 = conf_int[1] -0.008194 + 1.690067 * x_vals
    plt.plot(x_vals, y_vals1, '--r', label='Bottom Conf Int')
    plt.plot(x_vals, y_vals2, '--r', label='Top Conf Int')
    plt.plot(x_vals, y_vals, '-b', label='Best Fit')
    plt.xlabel("S&P500", fontsize=12)
    plt.ylabel("US Firm", fontsize=12)
    plt.legend()
    plt.savefig("conf_intervals1.png", dpi=400)
    # plt.savefig("conf_intervals_2_new_std.png", dpi=400)
    plt.close()
    

def prob2():
    axes = plt.gca()
    gdp = pd.read_csv("okun.csv").set_index("DATE")
    gdp['ue_diff'] = gdp['UNEMPLOYMENT'].diff(1)
    gdp['gdp_diff'] = gdp['GDP'].diff(1)
    # gdp.plot.scatter(x="ue_diff", y="gdp_diff")
    # plt.savefig("gdp_ue_scatter.png", dpi=400)
    # plt.close()

    gdp.plot.scatter(x="ue_diff", y="gdp_diff")
    reg = sm.ols(formula="gdp_diff ~ ue_diff", data=gdp).fit()
    beta = reg.params['ue_diff']
    alpha = reg.params['Intercept']
    x_vals = np.linspace(-4,4,100)
    
    y_vals = alpha + beta * x_vals
    # conf_int = scipy_int.norm.interval(0.95, loc=gdp['gdp_diff'].mean(), scale=gdp['gdp_diff'].std())
    # y_valsb = alpha + beta * x_vals + gdp['gdp_diff'].mean() + conf_int[0]
    # y_valst = alpha + beta * x_vals + gdp['gdp_diff'].mean() + conf_int[1]
    
    pred = reg.get_prediction(pd.DataFrame(x_vals, columns=['ue_diff']))
    y_valsb = pred.summary_frame(alpha=0.05)['obs_ci_lower']
    y_valst = pred.summary_frame(alpha=0.05)['obs_ci_upper']
    
    # t_val for 57 degrees of freedom
    # pdb.set_trace()
    # t_val = 2.0025
    # sse = (reg.resid.values**2).sum() / 57
    # s = np.sqrt(sse / (59-2))
    # xs = gdp['ue_diff'].var()
    # x_valst = np.array([pred_s(s, 59, gdp['ue_diff'].mean(), x, xs) for x in x_vals])
    # pdb.set_trace()
    # y_valsb = alpha + beta * x_vals - x_valst * t_val
    # y_valst = alpha + beta * x_vals + x_valst * t_val
    
    plt.plot(x_vals, y_valsb, '--r', label='Bottom Conf Int')
    plt.plot(x_vals, y_valst, '--r', label='Top Conf Int')
    plt.plot(x_vals, y_vals, '-b', label='Best Fit')
    # pd.DataFrame(reg.resid).reset_index().plot.scatter(x="DATE", y=0)
    # plt.ylabel("Residual", fontsize=12)
    # plt.savefig("resid_scatter.png", dpi=400)
    # plt.close()
    plt.legend()
    plt.savefig("gdp_pred.png", dpi=400)
    plt.close()
    
    
def prob3():
    lev = pd.read_csv("leverage.csv").set_index("Date")
    lev['spx_ret'] = lev['SPX'].pct_change()
    lev['vix_ret'] = lev['VIX'].pct_change()
    lev = lev.dropna()
    lev.plot.scatter(x="vix_ret", y="spx_ret")
    plt.savefig("vix_spx_scatter.png", dpi=400)
    plt.close()
    
    lev2 = lev[['spx_ret', 'vix_ret']]
    reg = sm.ols(formula="spx_ret ~ vix_ret", data=lev2).fit()
    anova = stats.stats.anova_lm(reg, typ=2)
    beta = reg.params['vix_ret']
    alpha = reg.params['Intercept']
    ssr = anova.loc['vix_ret']['sum_sq']
    sse = anova.loc['Residual']['sum_sq']
    s = np.sqrt(sse/(len(lev2)-2))
    sst = ssr + sse
    pdb.set_trace()
    t_val = 1.6479
    pre = pred_s(s, len(lev2)-2, lev2['vix_ret'].mean(), 0.1, lev2['vix_ret'].var())

    axes = plt.gca()
    axes.set_xlim([-0.2, 0.])
    # axes.set_xlim([-0.5, 1.5])
    axes.set_ylim([-0.05, 0.05])
    x_vals = np.linspace(axes.get_xlim()[0], axes.get_xlim()[1], 200)
    # y_vals = alpha + beta * x_vals
    # y_vals1 = conf_int[0] + alpha + beta * x_vals
    # y_vals2 = conf_int[1] + alpha + beta * x_vals
    
    # t_val for 57 degrees of freedom
    # t_val = 2.0025
    # sse = (reg.resid.values**2).sum() / 57
    # s = np.sqrt(sse / (59-2))
    # xs = lev2['vix_ret'].var()
    # x_valst = [pred_s(s, 59, gdp['ue_diff'].mean(), x, xs) for x in x_vals]
    # pdb.set_trace()
    # y_valsb = alpha + beta * x_vals - x_valst * t_val
    # y_valst = alpha + beta * x_vals + x_valst * t_val

    y_vals = alpha + beta * x_vals
    pred = reg.get_prediction(pd.DataFrame(x_vals, columns=['vix_ret']))
    y_valsb = pred.summary_frame(alpha=0.10)['obs_ci_lower']
    y_valst = pred.summary_frame(alpha=0.10)['obs_ci_upper']
    
    pdb.set_trace()
    # lev2.plot.scatter(x="vix_ret", y="spx_ret")
    plt.plot(x_vals, y_valsb, '--r', label='Bottom Conf Int')
    plt.plot(x_vals, y_valst, '--r', label='Top Conf Int')
    plt.plot(x_vals, y_vals, '-b', label='Best Fit')
    plt.xlabel("VIX", fontsize=12)
    plt.ylabel("SPX", fontsize=12)
    plt.legend()
    # plt.savefig("pred_intervals_vix.png", dpi=400)
    plt.savefig("pred_intervals_vix_noplot.png", dpi=400)
    # plt.savefig("conf_intervals_2_new_std.png", dpi=400)
    plt.close()    


def pred_s(s, n, xm, xf, xs):
    front = 1 + 1/n
    back = ((xf - xm)**2) / ((n - 1) * xs)
    sqrt = np.sqrt(front + back)
    tot = s * sqrt
    return tot


# prob2()
# prob1()
prob3()
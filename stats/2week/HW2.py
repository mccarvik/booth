"""
HW2 for Applied Regression
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
    """
    problem 1 of the HW
    """
    np.random.seed(6)
    xvals = np.random.normal(-1.0, 2.5, 100)
    errs = np.random.normal(0, 3, 100)
    yvals = 2.5 + 2*xvals + errs
    xy = pd.DataFrame(data=np.hstack((xvals[:,None], yvals[:,None])), columns=['x','y'])
    # sns.lmplot(x='x',y='y',data=xy,fit_reg=True, legend_out=False)
    xy.plot.scatter(x="x", y="y")
    coef = np.polyfit(xvals, yvals,1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(xvals, poly1d_fn(xvals), '-b', label='Orig')
    plt.savefig("./xy_scatter.png", dpi=400)
    
    coef = np.polyfit(xvals[:25], yvals[:25],1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(xvals, poly1d_fn(xvals), '-r', label='25')
    coef = np.polyfit(xvals[25:], yvals[25:],1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(xvals, poly1d_fn(xvals), '-g', label='75')
    plt.legend()
    plt.savefig("./xy_scatter_25_75.png", dpi=400)
    plt.close()

    # part iii
    xy_df = pd.DataFrame(data=[xvals, yvals]).T
    xy_df.columns = ['x', 'y']
    
    # full sample
    reg = sm.ols(formula="y ~ x", data=xy_df).fit()
    beta_main = reg.params['x']
    alpha_main = reg.params['Intercept']
    xy_mean = (xy['x']*beta_main + alpha_main).mean()
    print("alpha: " + str(alpha_main))
    print("beta: " + str(beta_main))
    print("mean 100: " + str(xy_mean))
    
    # 25 sample
    reg = sm.ols(formula="y ~ x", data=xy_df[:25]).fit()
    beta = reg.params['x']
    alpha = reg.params['Intercept']
    xy_mean_25 = (xy[:25]['x']*beta + alpha).mean()
    print("alpha: " + str(alpha))
    print("beta: " + str(beta))
    print("mean 25: " + str(xy_mean_25))

    # 75 sample
    reg = sm.ols(formula="y ~ x", data=xy_df[:75]).fit()
    beta = reg.params['x']
    alpha = reg.params['Intercept']
    xy_mean_75 = (xy[:75]['x']*beta + alpha).mean()
    print("alpha: " + str(alpha))
    print("beta: " + str(beta))
    print("mean 75: " + str(xy_mean_75))
    
    axes = plt.gca()
    conf_int = scipy_int.norm.interval(0.85, loc=-1, scale=2.5)
    xy.plot.scatter(x="x", y="y")
    coef = np.polyfit(xvals, yvals,1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(xvals, poly1d_fn(xvals), '-b', label='best_fit')
    x_vals = np.linspace(-8,8,100)
    y_vals = alpha_main + beta_main * x_vals
    # adjust by 1 for the mean of the expected X return
    y_vals1 = conf_int[0] + 1  + 2.5 + 2 * x_vals
    y_vals2 = conf_int[1] + 1 + 2.5 + 2 * x_vals
    plt.plot(x_vals, y_vals1, '--r', label='Bottom Conf Int')
    plt.plot(x_vals, y_vals2, '--r', label='Top Conf Int')
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.legend()
    # plt.savefig("conf_intervals.png", dpi=400)
    plt.savefig("conf_intervals_true.png", dpi=400)
    plt.close()
    
    # calc vals outside
    outside_hi = []
    outside_lo = []
    for ix, obs in xy_df.iterrows():
        # adjust by 1 for the mean of the expected X return
        bottom = conf_int[0] + 1 + 2.5 + 2.0 * obs['x']
        top = conf_int[1] + 1 + 2.5 + 2.0 * obs['x']
        if obs['y'] > top:
            outside_hi.append(obs)
        if obs['y'] < bottom:
            outside_lo.append(obs)
    
    print(len(outside_hi))
    print(len(outside_lo))

    
def prob2():
    lev = pd.read_csv("leverage.csv").set_index("Date")
    lev['spx_ret'] = lev['SPX'].pct_change()
    lev['vix_ret'] = lev['VIX'].pct_change()
    lev = lev.dropna()
    lev.plot.scatter(x="vix_ret", y="spx_ret")
    plt.savefig("vix_spx_scatter.png", dpi=400)
    plt.close()
    
    lev2 = lev[['spx_ret', 'vix_ret']]
    reg = linear_model.LinearRegression()
    reg.fit(lev['vix_ret'].values.reshape(-1,1), lev['spx_ret'].values)
    print(reg.coef_)
    
    # pdb.set_trace()
    # reg2 = sm.ols(formula="spx_ret ~ vix_ret", data=lev2).fit()
    
    sns.lmplot(x='vix_ret',y='spx_ret',data=lev,fit_reg=True)
    plt.savefig("./vix_spx_regr.png", dpi=400)
    plt.close()

    model = sm.ols(formula="spx_ret ~ vix_ret", data=lev2).fit()
    anova_ac = stats.stats.anova_lm(model, typ=2)
    ssr = anova_ac.loc['vix_ret']['sum_sq']
    sse = anova_ac.loc['Residual']['sum_sq']
    sst = ssr + sse
    print("sst:  {}    ssr:  {}   sse:  {}".format(sst, ssr, sse))
    print("ssr / sst:  {}".format(str(ssr / sst)))
    print(anova_ac)
    
    conf_int = scipy_int.norm.interval(0.9, loc=0, scale=1)
    conf_int = np.array(conf_int) * (0.025)
    # conf_int = np.array(conf_int) * (0.0094)
    axes = plt.gca()
    lev.plot.scatter(x="vix_ret", y="spx_ret")
    # axes.set_xlim([40, 80])
    # axes.set_ylim([12, 20])
    axes.set_xlim([-0.2, 1.2])
    axes.set_ylim([-0.1, 0.1])
    x_vals = np.array(axes.get_xlim())
    y_vals = 0.0076 + -0.0747 * x_vals
    y_vals1 = conf_int[0] + 0.0076 + -0.0747 * x_vals
    y_vals2 = conf_int[1] + 0.0076 + -0.0747 * x_vals
    plt.plot(x_vals, y_vals1, '--r', label='Bottom Conf Int')
    plt.plot(x_vals, y_vals2, '--r', label='Top Conf Int')
    plt.plot(x_vals, y_vals, '-b', label='Best Fit')
    plt.xlabel("VIX", fontsize=12)
    plt.ylabel("SPX", fontsize=12)
    plt.legend()
    plt.savefig("conf_intervals_2.png", dpi=400)
    # plt.savefig("conf_intervals_2_new_std.png", dpi=400)
    plt.close()
    

def prob3():
    van = pd.read_csv("vanguard.csv").set_index("Date")
    etfs = van.columns[:-2]
    
    # get Excess_ret
    excess_r = ((np.log(van["SPX INDEX"]) - np.log(van['SPX INDEX'].shift(1))) * 52) - (van['TBILL']/100)
    
    axes = plt.gca()
    scatter = []
    for etf in etfs:
        # get annual returns
        log_ret = ((np.log(van[etf]) - np.log(van[etf].shift(1))) * 52) - (van['TBILL']/100)
        ret_df = pd.concat([log_ret[1:], excess_r[1:]], axis=1)
        ret_df.columns = ['ret', 'mkt']
        reg = sm.ols(formula="ret ~ mkt", data=ret_df).fit()
        
        # reg = linear_model.LinearRegression()
        # reg.fit(log_ret[1:].values.reshape(-1, 1), excess_r[1:].values)
        # pdb.set_trace()
        beta = reg.params['mkt']
        alpha = reg.params['Intercept']
        plt.plot(alpha, beta, "o", color='blue')
        axes.annotate(etf, (alpha, beta))
        scatter.append([etf, alpha, beta])
    
    pdb.set_trace()
    ab = pd.DataFrame(data=scatter, columns=["ticker", "alpha", "beta"]).set_index("ticker").to_csv("vanguard_alpha_beta.csv")
    axes.set_xlim([-0.21, 0.16])
    axes.set_ylim([-0.5, 1.25])
    plt.xlabel("Alpha", fontsize=12)
    plt.ylabel("Beta", fontsize=12)
    plt.savefig("alpha_beta_scatter.png",dpi=400)
    plt.close()
    
    
prob1()
# prob2()
# prob3()
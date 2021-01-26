"""
HW5 for Applied Regression
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


def prob11(twts):
    # ANOVA analysis for 
    model = sm.ols(formula= "favoriteCount ~ source", data=twts).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    ssr = anova.loc['source']['sum_sq']
    sse = anova.loc['Residual']['sum_sq']
    sst = ssr + sse
    print("FavoriteCount")
    print("sst:  {}    ssr:  {}   sse:  {}".format(round(sst,2), round(ssr,2), round(sse,2)))
    print("ssr / sst (R^2):  {}".format(str(round(ssr / sst,2))))
    print(anova)
    twts[['favoriteCount', 'source']].boxplot(by='source')
    plt.savefig("./boxplot_fav_source.png", dpi=400)
    plt.close()
    
    # ANOVA analysis for 
    model = sm.ols(formula= "retweetCount ~ source", data=twts).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    ssr = anova.loc['source']['sum_sq']
    sse = anova.loc['Residual']['sum_sq']
    sst = ssr + sse
    print("RetweetCount")
    print("sst:  {}    ssr:  {}   sse:  {}".format(round(sst,2), round(ssr,2), round(sse,2)))
    print("ssr / sst (R^2):  {}".format(str(round(ssr / sst,2))))
    print(anova)
    twts[['retweetCount', 'source']].boxplot(by='source')
    plt.savefig("./boxplot_retweet_source.png", dpi=400)
    plt.close()
    
    
def prob12(twts):
    excl = twts[['exclamation', 'source']]
    print(excl.value_counts())
    print(excl.value_counts(normalize=True))
    excl["s_d"] = excl.apply(lambda x: 1 if x['source'] == 'Android' else 0, axis=1)
    excl["e_d"] = excl.apply(lambda x: 1 if x['exclamation'] == 'Yes' else 0, axis=1)
    print(excl[['s_d', 'e_d']].corr())
    model = sm.ols(formula= "e_d ~ s_d", data=excl).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    ssr = anova.loc['s_d']['sum_sq']
    sse = anova.loc['Residual']['sum_sq']
    sst = ssr + sse
    print("ssr / sst (R^2):  {}".format(str(round(ssr / sst,2))))
    print(anova)
    
    pdb.set_trace()
    hsht = twts[['hashtag', 'source']]
    print(hsht.value_counts())
    print(hsht.value_counts(normalize=True))
    hsht["s_d"] = hsht.apply(lambda x: 1 if x['source'] == 'Android' else 0, axis=1)
    hsht["h_d"] = hsht.apply(lambda x: 1 if x['hashtag'] == 'Yes' else 0, axis=1)
    print(hsht[['s_d', 'h_d']].corr())
    model = sm.ols(formula= "h_d ~ s_d", data=hsht).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    ssr = anova.loc['s_d']['sum_sq']
    sse = anova.loc['Residual']['sum_sq']
    sst = ssr + sse
    print("ssr / sst (R^2):  {}".format(str(round(ssr / sst,2))))
    print(anova)
    

def prob13(twts):
    for factor in ['picture', 'exclamation', 'hashtag', 'hour']:
        # ANOVA analysis for
        model = sm.ols(formula= "favoriteCount ~ {}".format(factor), data=twts).fit()
        anova = stats.stats.anova_lm(model, typ=2)
        ssr = anova.loc[factor]['sum_sq']
        sse = anova.loc['Residual']['sum_sq']
        sst = ssr + sse
        print("FavoriteCount")
        print("sst:  {}    ssr:  {}   sse:  {}".format(round(sst,2), round(ssr,2), round(sse,2)))
        print("ssr / sst (R^2):  {}".format(str(round(ssr / sst,2))))
        print(anova)
        twts[['favoriteCount', factor]].boxplot(by=factor)
        plt.savefig("./boxplot_fav_{}.png".format(factor), dpi=400)
        plt.close()
        
        # ANOVA analysis for 
        model = sm.ols(formula= "retweetCount ~ {}".format(factor), data=twts).fit()
        anova = stats.stats.anova_lm(model, typ=2)
        ssr = anova.loc[factor]['sum_sq']
        sse = anova.loc['Residual']['sum_sq']
        sst = ssr + sse
        print("RetweetCount")
        print("sst:  {}    ssr:  {}   sse:  {}".format(round(sst,2), round(ssr,2), round(sse,2)))
        print("ssr / sst (R^2):  {}".format(str(round(ssr / sst,2))))
        print(anova)
        twts[['retweetCount', factor]].boxplot(by=factor)
        plt.savefig("./boxplot_retweet_{}.png".format(factor), dpi=400)
        plt.close()


def prob13b(twts):
    # twts['excl_hash'] = twts['exclamation'] and twts['hashtag']
    twts['excl_hash'] = twts.apply(lambda x: 1 if x['exclamation']=="Yes" and x['hashtag'] == "Yes" else 0, axis=1)
    twts['hr_bucket'] = twts.apply(lambda x: add_hr_bucket(x['hour']), axis=1)
    twts['hr1'] = twts.apply(lambda x: 1 if x['hr_bucket'] == 0 else 0, axis=1)
    twts['hr2'] = twts.apply(lambda x: 1 if x['hr_bucket'] == 1 else 0, axis=1)
    twts['hr3'] = twts.apply(lambda x: 1 if x['hr_bucket'] == 2 else 0, axis=1)
    twts['hr4'] = twts.apply(lambda x: 1 if x['hr_bucket'] == 3 else 0, axis=1)
    
    model = sm.ols(formula= "favoriteCount ~ exclamation + hashtag + exclamation*hashtag", data=twts).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    # print(model.summary())
    # print(anova)
    
    model = sm.ols(formula= "retweetCount ~ exclamation + hashtag + exclamation*hashtag", data=twts).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    # print(model.summary())
    # print(anova)
    
    # adding hr bucket
    model = sm.ols(formula= "favoriteCount ~ exclamation + hashtag + picture + hr1 + hr2 + hr3 + hr4 + exclamation*hashtag", data=twts).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    # print(model.summary())
    # print(anova)
    
    model = sm.ols(formula= "retweetCount ~ exclamation + hashtag + picture + hr1 + hr2 + hr3 + hr4 + exclamation*hashtag", data=twts).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    # print(model.summary())
    # print(anova)
    
    model = sm.ols(formula= "retweetCount ~ exclamation*source + hashtag*source + picture*source + hr1*source + hr2*source + hr3*source + hr4*source", data=twts).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    # print(model.summary())
    # print(anova)
    
    data = twts[twts['source'] == 'iPhone']
    model = sm.ols(formula= "favoriteCount ~ exclamation + hashtag + picture + hr1 + hr2 + hr3 + hr4", data=data).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    print(model.summary())
    print(anova)
    betas = model.params.to_frame()
    betas.columns = ['iPhone']
    betas['iPhone stderr'] = model.bse
    
    data = twts[twts['source'] == 'Android']
    model = sm.ols(formula= "favoriteCount ~ exclamation + hashtag + picture + hr1 + hr2 + hr3 + hr4", data=data).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    print(model.summary())
    print(anova)
    betas2 = model.params.to_frame()
    betas2.columns = ['Android']
    betas['Android'] = betas2['Android']
    betas['Beta Diff'] = betas['iPhone'] - betas['Android']
    betas['iPhone + 2*stderr'] = betas['iPhone'] + betas['iPhone stderr'] * 2
    betas['iPhone - 2*stderr'] = betas['iPhone'] - betas['iPhone stderr'] * 2
    betas['ACCEPT NULL'] = betas.apply(lambda x: x['Android'] <  x['iPhone + 2*stderr'] and x['Android'] > x['iPhone - 2*stderr'], axis=1)
    pdb.set_trace()
    print(betas)

    data = twts[twts['source'] == 'iPhone']
    model = sm.ols(formula= "retweetCount ~ exclamation + hashtag + picture + hr1 + hr2 + hr3 + hr4", data=data).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    print(model.summary())
    print(anova)
    betas = model.params.to_frame()
    betas.columns = ['iPhone']
    betas['iPhone stderr'] = model.bse
    
    data = twts[twts['source'] == 'Android']
    model = sm.ols(formula= "retweetCount ~ exclamation + hashtag + picture + hr1 + hr2 + hr3 + hr4", data=data).fit()
    anova = stats.stats.anova_lm(model, typ=2)
    print(model.summary())
    print(anova)
    betas2 = model.params.to_frame()
    betas2.columns = ['Android']    
    betas['Android'] = betas2['Android']
    betas['Beta Diff'] = betas['iPhone'] - betas['Android']
    betas['iPhone + 2*stderr'] = betas['iPhone'] + betas['iPhone stderr'] * 2
    betas['iPhone - 2*stderr'] = betas['iPhone'] - betas['iPhone stderr'] * 2
    betas['ACCEPT NULL'] = betas.apply(lambda x: x['Android'] <  x['iPhone + 2*stderr'] and x['Android'] > x['iPhone - 2*stderr'], axis=1)
    pdb.set_trace()
    print(betas)
    
    
    return
    for ind in ['favoriteCount', 'retweetCount']:
        pdb.set_trace()
        print(ind)
        res_df = pd.DataFrame()
        sb = {}
        for src in ["Android", "iPhone"]:
            data = twts[twts['source'] == src]
            n = len(data)
            n_pop = len(twts)
            model = sm.ols(formula= "{} ~ exclamation + hashtag + picture + hr1 + hr2 + hr3 + hr4 + exclamation*hashtag".format(ind), data=data).fit()
            summ = (model.resid**2).sum()
            print("resid^2 sum:  {}".format(summ))
            df = 9
            print ("DF: 9")
            print("n pop:  {}".format(n_pop))
            print("n:  {}".format(n))
            s2 = (1 / (n-df)) * summ
            print("s2 =  {}".format(s2))
            sx2 = data[ind].std()**2
            print ("sx2 =  {}".format(sx2))
            sb[src] = (s2 / ((n - 1) * sx2))**0.5
            print("sb =  {}".format(sb[src]))
            betas = model.params.to_frame()
            betas.columns = [src]
            if res_df.empty:
                res_df = betas
            else:
                res_df = pd.merge(res_df, betas, left_index=True, right_index=True)
        print(res_df[['Android', 'iPhone']])
        sdiff = (sb['Android']**2 + sb['iPhone']**2)**0.5
        print("sdiff =  {}".format(sdiff))
        res_df['Diff'] = res_df['Android'] - res_df['iPhone']
        res_df['Diff / s'] = res_df['Diff'] / sdiff
        res_df['Accept Null'] = abs(res_df['Diff / s']) < 1.96
        print(res_df)



def add_hr_bucket(hour):
    if hour < 6:
        return 0
    elif hour < 12:
        return 1
    elif hour < 18:
        return 2
    else:
        return 3



twts = pd.read_csv("trump.csv")
# prob11(twts)
# prob12(twts)
# prob13(twts)
prob13b(twts)
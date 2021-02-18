"""
HW5 ML
"""

import pdb
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.linear_model import Lasso
import scipy.stats as scipy
from sklearn.ensemble import GradientBoostingRegressor

LOTS = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add',
       'Mas_Vnr_Area', 'BsmtFin_SF_1', 'BsmtFin_SF_2', 'Bsmt_Unf_SF',
       'First_Flr_SF', 'Bsmt_Full_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr', 'Kitchen_AbvGr',
       'TotRms_AbvGrd', 'Fireplaces', 'Garage_Area', 'Wood_Deck_SF',
       'Open_Porch_SF', 'Enclosed_Porch', 'Screen_Porch', 'Mo_Sold', 'Year_Sold', 'Longitude', 'Latitude',
       'MS_SubClass_other', 'MS_Zoning_Residential_Medium_Density',
       'MS_Zoning_other', 'Lot_Shape_Slightly_Irregular','Land_Slope_other', 'Condition_1_other',
       'Overall_Qual_Above_Average', 'Overall_Qual_Good', 'Overall_Qual_Very_Good', 'Overall_Qual_other',
       'Overall_Cond_Above_Average', 'Overall_Cond_Good', 'Overall_Cond_other',
       'Roof_Style_Hip', 'Exter_Qual_Typical', 'Exter_Qual_other', 'Exter_Cond_other',
       'Bsmt_Qual_other', 'Bsmt_Cond_other', 'Bsmt_Exposure_No',
       'BsmtFin_Type_1_GLQ', 'Heating_QC_Typical', 'Central_Air_other',
       'Kitchen_Qual_Typical', 'Kitchen_Qual_other', 'Functional_other', 'Garage_Type_Detchd', 'Paved_Drive_other',
       'Sale_Type_other', 'Sale_Condition_other', 'Gr_Liv_Area']


SKIP = ['logsale', 'BsmtFin_SF_2', 'Bsmt_Full_Bath', 'Kitchen_AbvGr', 'Mas_Vnr_Area', 'Full_Bath',
        'Half_Bath', 'Bedroom_AbvGr', 'TotRms_AbvGrd', 'Fireplaces', 'Wood_Deck_SF', 'Open_Porch_SF',
        'Enclosed_Porch', 'Screen_Porch', 'MS_Zoning_other', 'Land_Slope_other', 'Overall_Qual_Very_Good',
        'Exter_Qual_other', 'Exter_Cond_other', 'Bsmt_Qual_other', 'Bsmt_Cond_other', 'Central_Air_other'
        'Kitchen_Qual_other', 'Functional_other', 'Paved_Drive_other', 'Central_Air_other', 'Kitchen_Qual_other']


LESS = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add',
       'Mas_Vnr_Area', 'BsmtFin_SF_1', 'Bsmt_Unf_SF',
       'First_Flr_SF', 'Bsmt_Full_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr',
       'TotRms_AbvGrd', 'Fireplaces', 'Garage_Area', 'Wood_Deck_SF',
       'Open_Porch_SF', 'Enclosed_Porch', 'Mo_Sold', 'Longitude', 'Latitude',
       'Overall_Qual_Good', 'Overall_Qual_Very_Good', 'Overall_Qual_other',
       'Overall_Cond_other', 'Exter_Qual_Typical',
       'Bsmt_Qual_other', 'BsmtFin_Type_1_GLQ', 'Central_Air_other',
       'Kitchen_Qual_Typical', 'Sale_Condition_other', 'Gr_Liv_Area']


def get_data():
    train = pd.read_csv("housing_train.csv")
    test = pd.read_csv("housing_test.csv")
    return train, test


def q21(train, test):
    """
    """
    train['logsale'] = np.log(train['Sale_Price'])
    # pdb.set_trace()
    # train['Sale_Price'].hist()
    # plt.savefig("sale_hist.png", dpi=300)
    # plt.close()
    # train['logsale'].hist()
    # plt.savefig("logsale_hist.png", dpi=300)
    # plt.close()
    # plt.scatter(train['Gr_Liv_Area'], train['Sale_Price'])
    # plt.savefig("sale_scatter.png", dpi=300)
    # plt.close()
    plt.scatter(train['Gr_Liv_Area'], train['logsale'])
    plt.savefig("logsale_scatter.png", dpi=300)
    plt.close()


def q22(train, test):
    """
    """
    train['logsale'] = np.log(train['Sale_Price'])
    bic_pred = {}
    mse_pred = {}
    
    # forwards selection
    covars = train.drop(columns=['Sale_Price', 'logsale']).columns
    cols = []
    formbic = 100000000000000
    formula = "logsale ~ "
    print("BIC")
    while len(covars):
        minbic = 10000000000000
        if len(formula) > 10:
            formula = formula + " + "
        for cov in covars:
            if cov in cols:
                continue
            reg = sm.ols(formula=formula + cov, data=train).fit()
            if reg.bic < minbic:
                minbic = reg.bic
                mincov = cov
        if minbic > formbic:
            break
        formula = formula + mincov
        formbic = minbic
        cols.append(mincov)
        reg = sm.ols(formula=formula, data=train).fit()
        print("added: " + mincov + "     BIC: " +  str(formbic))
        bic_pred[len(cols)] = formbic
    print("Final Columns:  {}".format(str(len(cols))))
    print(cols)
    print("Final BIC:")
    print(formbic)
    
    
    covars = train.drop(columns=['Sale_Price', 'logsale']).columns
    cols = []
    formbic = 100000000000000
    print("CV")
    while len(covars):
        minbic = 10000000000000
        if len(formula) > 10:
            formula = formula + " + "
        for cov in covars:
            if cov in cols:
                continue
            y = train['logsale']
            x = np.array(train[cols + [cov]])
            reg = LinearRegression().fit(x, y)
            score = abs(cross_val_score(reg, x, y, cv=5, scoring='neg_mean_squared_error').mean())
            std = cross_val_score(reg, x, y, cv=5, scoring='neg_mean_squared_error').std()
            if score < minbic:
                minbic = score
                mincov = cov
        if minbic > formbic:
            break
        formula = formula + mincov
        formbic = minbic
        cols.append(mincov)
        y = train['logsale']
        x = np.array(train[cols + [cov]])
        reg = LinearRegression().fit(x, y)
        score = abs(cross_val_score(reg, x, y, cv=5, scoring='neg_mean_squared_error').mean())
        std = cross_val_score(reg, x, y, cv=5, scoring='neg_mean_squared_error').std()
        print("added: " + mincov + "     MSE: " +  str(round(formbic, 5)) + "     StDev: " + str(round(std, 5)))
        mse_pred[len(cols)] = formbic
    print("Final Columns:  {}".format(str(len(cols))))
    print(cols)
    print("Final MSE:")
    print(formbic)
    
    fig,ax = plt.subplots()
    lns1 = ax.plot(bic_pred.keys(), bic_pred.values(), c='b', label="BIC")
    ax.set_xlabel("predictors")
    ax.set_ylabel("BIC")
    # twin object for two different y-axis on the sample plot
    ax2=ax.twinx()
    # make a plot with different y-axis using second axis object
    lns2 = ax2.plot(mse_pred.keys(), mse_pred.values(), c='r', label="MSE")
    ax2.set_ylabel("MSE")
    # added these three lines
    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    plt.savefig("bic_mse.png", dpi=300)
    plt.close()


def q23(train, test):
    """
    Lasso
    """
    train['logsale'] = np.log(train['Sale_Price'])
    covars = train.drop(columns=['Sale_Price', 'logsale']).columns
    
    # Original Model
    scores = []
    alphas = []
    coefs = []
    stds = []
    coef_vals = {}
    final_coefs = []
    for alpha in range(-4,14,1):
        clf = Lasso(alpha=10**(-1*alpha/2), random_state=0, max_iter=10000)
        score = cross_val_score(clf, train[covars], train['logsale'], cv=5, scoring='neg_mean_squared_error').mean()
        std = cross_val_score(clf, train[covars], train['logsale'], cv=5, scoring='neg_mean_squared_error').std()
        clf.fit(train[covars], train['logsale'])
        coef = len([i for i in clf.coef_ if np.abs(i) > 0])
        if alpha == -2:
            final_coefs = clf.coef_
        coef_vals[-1*alpha/2] = clf.coef_
        scores.append(-1*score)
        alphas.append(-1*alpha/2)
        coefs.append(coef)
        stds.append(std)
        # score = cross_val_score(clf, data[0], data[2], cv=5, scoring='accuracy').mean()
        
    for score, alpha, std, cf in zip(scores, alphas, stds, coefs):
        print("Lambda:  10^{}  Coefs:  {}  StDev:  {}   MSE:  {}".format(str(alpha), str(cf), str(std), str(score)))

    plt.plot(alphas, scores, "-", c="b", label="mse")
    plt.legend()
    plt.savefig("mse_cv_lasso" + '.png', dpi=300)
    plt.close()
    
    cf = pd.DataFrame(coef_vals)
    alphas = list(range(-4,14,1))
    
    for x in range(len(cf)):
        plt.plot([a*-0.5 for a in alphas], cf.loc[x], "-")
    plt.savefig("coefs" + '.png', dpi=300)
    plt.close()
    
    # cft = cf.T
    # cft.columns = train[covars].columns
    # feats = cft.loc[-3.5]
    # feats = dict(sorted(zip(train[covars].columns, feats), key=lambda item: -abs(item[1])))
    # for k, v in feats.items():
    #     print(k + ":  " + str(v))


def dataclean(train, test):
    # Remove odd data
    train = train.drop([1089])
    # train_t = train[['logsale', 'Gr_Liv_Area']]
    # train_t['diff'] = train['logsale'] - train['Gr_Liv_Area']
    # train_t = train_t[['diff']]
    # zs = scipy.zscore(train_t)
    # abs_z_scores = np.abs(zs)
    # filtered_entries = (abs_z_scores < 4).all(axis=1)
    # train = train[filtered_entries]

    train = train.drop(columns=['Sale_Price'])
    return train, test


def outliers(train, y):
    # Remove odd data
    train['logsale'] = y
    for i in train.columns:
        if i in SKIP:
            continue
        train_t = train[[i]]
        zs = scipy.zscore(train_t)
        abs_z_scores = np.abs(zs)
        filtered_entries = (abs_z_scores < 5).all(axis=1)
        if abs(len(train[filtered_entries]) - len(train))> 0:
            # plt.scatter(train[i], train['logsale'])
            # plt.savefig("test.png")
            # plt.close()
            print("filtered  {}  {}".format(i, str(abs(len(train[filtered_entries]) - len(train)))))
            train = train[filtered_entries]
    y = train['logsale']
    train = train.drop(columns=["logsale"])
    return train, y


def q25(train, test):
    train['logsale'] = np.log(train['Sale_Price'])
    train, test = dataclean(train, test)
    
    ALL = train.drop(columns=['logsale']).columns
    # x_train = train
    # x_train, x_test, _, _ = train_test_split(train, train['logsale'], test_size=0.25, random_state=1)
    x_train, x_test, _, _ = train_test_split(train, train['logsale'], test_size=0.15)
    # feats = train.drop(columns=["logsale"]).columns
    feats = ['Lot_Frontage', 'Lot_Area', 'Year_Built', 'Year_Remod_Add',
       'Mas_Vnr_Area', 'BsmtFin_SF_1', 'Bsmt_Unf_SF',
       'First_Flr_SF', 'Bsmt_Full_Bath', 'Full_Bath', 'Half_Bath', 'Bedroom_AbvGr',
       'TotRms_AbvGrd', 'Fireplaces', 'Garage_Area', 'Wood_Deck_SF',
       'Open_Porch_SF', 'Mo_Sold', 'Longitude', 'Latitude',
       'Overall_Qual_Very_Good', 'Overall_Qual_other',
       'Exter_Qual_Typical',
       'Bsmt_Qual_other', 'BsmtFin_Type_1_GLQ', 'Central_Air_other',
       'Kitchen_Qual_Typical', 'Sale_Condition_other', 'Gr_Liv_Area']
    
    # pdb.set_trace()
    # # get cross val score
    # for est in range(1, 40, 3):
    #     # rfc = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=1,
    #     #                                 max_leaf_nodes=65, min_samples_split=est, max_features=8)
    #     rfc = GradientBoostingRegressor(n_estimators=300, max_depth=5, max_leaf_nodes=52, max_features=est)
    #     scores = cross_val_score(rfc, work, train['logsale'], cv=5, scoring='neg_mean_squared_error')
    #     print("{}:  {}".format(est, str(scores.mean())))

    work = x_train[LOTS]
    y = x_train['logsale']
    # work, y = outliers(work, x_train['logsale'])
    rfc = GradientBoostingRegressor(n_estimators=1000, max_depth=5, max_leaf_nodes=12, max_features=20)
    rfc.fit(work, y)
    # y = x_train['logsale']
    # x = np.array(work)
    # rfc = LinearRegression().fit(x, y)
    mse = mean_squared_error(x_test['logsale'], rfc.predict(x_test[work.columns]))
    print("MSE:  {}".format(mse))
    
    feats = rfc.feature_importances_
    feats = dict(sorted(zip(work.columns, feats), key=lambda item: -item[1]))
    for k, v in feats.items():
        print
        print(k + ":  " + str(v))
    
    work = train[feats]
    test = test[work.columns]
    rfc.fit(work, train['logsale'])
    create_csv(rfc, test)

def create_csv(model, test):
    y_pred = pd.DataFrame(model.predict(test)).reset_index()
    y_pred.columns = ["Id", "Sale_Price"]
    y_pred['Id'] = y_pred['Id'] + 1
    y_pred = y_pred.set_index("Id")
    pdb.set_trace()
    y_pred['Sale_Price'] = np.exp(y_pred['Sale_Price'])
    # pdb.set_trace()
    # y_pred['Sale_Price'] = y_pred.apply(lambda x: 4 if x['count'] < 0 else x['count'], axis=1)
    y_pred.to_csv("sampleSubmission.csv")


if __name__ == '__main__':
    data = get_data()
    # q21(data[0], data[1])
    # q22(data[0], data[1])
    # q23(data[0], data[1])
    q25(data[0], data[1])
"""
HW4 ML
"""

import pdb
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from scikitplot.metrics import plot_lift_curve
from scikitplot.helpers import cumulative_gain_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
import scipy.stats as scipy


backward_cols = ['Early.RPL', 'Is.Non.Annual.', 'Num.of.Non_FPP.PAX', 'SPR.Group.Revenue', 'Tuition', 
                'MDR.High.Grade', 'DepartureMonth', 'DifferenceTraveltoFirstMeeting', 'FRP.Take.up.percent.', 
                'School.Sponsor', 'FirstMeeting', 'Group.State', 'GroupGradeType', 'FPP.to.School.enrollment', 
                'EZ.Pay.Take.Up.Rate', 'Total.Pax', 'SchoolGradeTypeLow', 'Region', 'To.Grade', 'MDR.Low.Grade', 
                'Total.School.Enrollment', 'Days', 'NumberOfMeetingswithParents', 'Initial.System.Date', 
                'SchoolSizeIndicator', 'FRP.Active', 'Poverty.Code', 'SingleGradeTripFlag', 'SPR.Product.Type', 
                'From.Grade', 'SchoolGradeTypeHigh', 'FPP.to.PAX', 'GroupGradeTypeHigh', 'SPR.New.Existing', 
                'Cancelled.Pax', 'SchoolGradeType', 'CRM.Segment', 'Income.Level', 'Total.Discount.Pax', 
                'Travel.Type', 'Deposit.Date', 'GroupGradeTypeLow']


FACTORS = ['To.Grade', 'Is.Non.Annual.', 'Days', 'CRM.Segment', 'Parent.Meeting.Flag', 'MDR.High.Grade', 
           'School.Sponsor', 'NumberOfMeetingswithParents', 'SingleGradeTripFlag',
           'Program.Code', 'From.Grade', 'Group.State', 'Travel.Type', 'Special.Pay', 'Poverty.Code',
           'Region', 'School.Type', 'MDR.Low.Grade', 'Income.Level', 'SPR.Product.Type', 'SPR.New.Existing',
           'SchoolGradeTypeLow', 'SchoolGradeTypeHigh', 'SchoolGradeType', 'DepartureMonth', 'GroupGradeTypeLow',
           'GroupGradeTypeHigh', 'GroupGradeType', 'MajorProgramCode', 'SchoolSizeIndicator']
DATES = ['Departure.Date', 'Return.Date', 'Deposit.Date', 'Early.RPL', 'Latest.RPL',
                'Initial.System.Date', 'FirstMeeting', 'LastMeeting', '>= 3 FPP Date','>= 10 FPP Date','>= 20 FPP Date','>= 35 FPP Date']


def cleandata(data):
    # Factor
    #     To.Grade, Is.Non.Annual., Days, CRM.Segment, Parent.Meeting.Flag,
    # MDR.High.Grade, School.Sponsor, NumberOfMeetingswithParents, SingleGradeTripFlag can be converted
    # to factors. We can also convert the output Retained.in.2012.
    
    factor_cat = ['To.Grade', 'Is.Non.Annual.', 'Days', 'CRM.Segment', 'Parent.Meeting.Flag',
                  'MDR.High.Grade', 'School.Sponsor', 'NumberOfMeetingswithParents',
                  'SingleGradeTripFlag', 'Retained.in.2012.']
    date_cat = ['Departure.Date', 'Return.Date', 'Deposit.Date', 'Early.RPL', 'Latest.RPL',
                'Initial.System.Date', 'FirstMeeting', 'LastMeeting',
                '>= 3 FPP Date','>= 10 FPP Date','>= 20 FPP Date','>= 35 FPP Date']
    
    combine = True
    if combine:
        data2 = pd.read_csv("travelData_supplement.csv")
        data = data.merge(data2)
    
    for cat in date_cat:
        data[cat] = pd.to_datetime(data[cat])
        
    for col in data.columns:
        if not data[col].isnull().values.any():
            print("{} is fine".format(col))
            continue
        
        data[col + "_surrogate"] = data[col].isnull()
        data[col + "_surrogate"] = data.apply(lambda x: 1 if x[col + "_surrogate"] else 0, axis=1)
        if data[col].dtype.kind in 'biufc':
            data[col] = data[col].fillna(0)
        elif col in date_cat:
            data[col] = data[col].fillna(dt.datetime(1900,1,1,0,0))
        else:
            data[col] = data[col].fillna("FIXED_NA")
        print("{} edited".format(col))
    
    
    # deal with other category
    for cat in data.columns:
        if (data[cat].dtype == object and isinstance(data.iloc[0][cat], str)) or cat in factor_cat:
            items = data[cat].value_counts().items()
            for val, ct in items:
                if ct > 10:
                    continue
                else:
                    data[cat] = data.apply(lambda x: "Other." + cat if x[cat]==val else x[cat], axis=1)
                    print("{} = {} switched to other".format(cat, val))
    
    if combine:
        data.to_csv("correctedCombinedData.csv")
    else:
        data.to_csv("correctedData.csv")
    
    pdb.set_trace()
    return data


def traintest(data):
    data = train_test_split(data.drop(columns=['Retained.in.2012.']), data['Retained.in.2012.'], test_size=0.209, random_state=5)
    return data


def get_correct_data():
    data = pd.read_csv("correctedData.csv")
    return data

def get_correct_combined():
    data = pd.read_csv("correctedCombinedData.csv")
    return data


def get_data():
    data = pd.read_csv("travelData.csv")
    return data


def factorize(data):
    for cat in FACTORS:
        data[cat] = pd.factorize(data[cat])[0]
    
    for cat in DATES:
        data[cat] = pd.to_datetime(data[cat]).astype('int64')
        max_a = data[cat].max()
        min_a = data[cat].min()
        min_norm = -1
        max_norm =1
        data[cat] = (data[cat] - min_a) *(max_norm - min_norm) / (max_a-min_a) + min_norm

    data = data.drop(columns=['Unnamed: 0', 'ID'])
    # drop surrogate columns
    data = data[data.columns[:-18]]
    return data


def q5(data):
    # Logistic regression
    # data 0 = x_train, 1 = x_test, 2 = y_train, 3 = y_tes
    
    # Original Model
    # clf = LogisticRegression(random_state=0, max_iter=10000).fit(data[0], data[2])
    # print("Train Accuracy:")
    # print(clf.score(data[0], data[2]))
    # print("Test Accuracy:")
    # print(accuracy_score(clf.predict(data[1]), data[3]))
    
    # backwards selection
    # removes surrogates
    # covars = data[0].columns
    # cols = []
    # FULL_BIC = 10000
    # pdb.set_trace()
    # while len(covars) > 1:
    #     min_bic = 0
    #     for cov in covars:
    #         temp_covs = list(set(covars) - set([cov]))
    #         temp_data = data[0][temp_covs]
    #         clf = LogisticRegression(random_state=0, max_iter=10000).fit(temp_data, data[2])
    #         # scores = cross_val_score(tree, x_train[temp_covs], y_train, cv=5, scoring='neg_mean_squared_error')
    #         bic = calculate_bic(temp_data, data[2], clf)
    #         # print("BIC:  {}  score:  {}  var rem:  {}".format(str(bic), str(clf.score(temp_data, data[2])), cov))
    #         if bic < min_bic:
    #             min_bic = bic
    #             maxcov = cov
    #     if min_bic > FULL_BIC:
    #         break
    #     else:
    #         FULL_BIC = min_bic
    #     covars = list(set(covars) - set([maxcov]))
    #     print("Removed: " + maxcov + "     BIC: " +  str(FULL_BIC))
    # print("Final Columns:")
    # print(covars)
    # print("Final BIC:")
    # print(FULL_BIC)
        
    # pdb.set_trace()
    # print()
    
    # forwards selection
    covars = data[0].columns
    cols = []
    FULL_BIC = 100000000000000
    pdb.set_trace()
    while len(covars):
        if len(cols) == 10:
            break
        min_bic = 10000000
        for cov in covars:
            temp_covs = cols + [cov]
            temp_data = data[0][temp_covs]
            clf = LogisticRegression(random_state=0, max_iter=10000).fit(temp_data, data[2])
            bic = calculate_bic(temp_data, data[2], clf)
            # print("BIC:  {}  score:  {}  var rem:  {}".format(str(bic), str(clf.score(temp_data, data[2])), cov))
            if bic < min_bic:
                min_bic = bic
                maxcov = cov
        if min_bic > FULL_BIC:
            # break
            print
        else:
            FULL_BIC = min_bic
        cols.append(maxcov)
        covars = list(set(covars) - set([maxcov]))
        print("added: " + maxcov + "     Accuracy: " + str(clf.score(temp_data, data[2])) + "     BIC: " +  str(FULL_BIC))
    
    print("Final Columns:")
    print(cols)
    print("Final BIC:")
    print(FULL_BIC)


def q5lasso(data):
    # Logistic regression
    # data 0 = x_train, 1 = x_test, 2 = y_train, 3 = y_tes
    
    # Original Model
    scores = []
    alphas = []
    coefs = []
    stds = []
    coef_vals = {}
    final_coefs = []
    # for alpha in range(-2,12,1):
    for alpha in range(-4,8,1):
        clf = Lasso(alpha=10**(-1*alpha/2), random_state=0, max_iter=10000)
        score = cross_val_score(clf, data[0], data[2], cv=5, scoring='neg_mean_squared_error').mean()
        std = cross_val_score(clf, data[0], data[2], cv=5, scoring='neg_mean_squared_error').std()
        clf.fit(data[0],data[2])
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
    alphas = list(range(-4,8,1))
    
    for x in range(len(cf)):
        plt.plot([a*-0.5 for a in alphas], cf.loc[x], "-")
    plt.savefig("coefs" + '.png', dpi=300)
    plt.close()
    
    cft = cf.T
    cft.columns = data[0].columns
    pdb.set_trace()
    print(cft.loc[-2])
    
    # clf = Lasso(alpha=10**(-2), random_state=0, max_iter=10000)


def q6(data):
    clf = Lasso(alpha=10**(-2), random_state=0, max_iter=10000)
    clf.fit(data[0],data[2])
    y_pred = [1 if a > 0.6072 else 0 for a in clf.predict(data[1])]
    yprobs1 = clf.predict(data[1])
    print("Train Accuracy:")
    print(accuracy_score([1 if a > 0.6072 else 0 for a in clf.predict(data[0])], data[2]))
    print("Test Accuracy:")
    print(accuracy_score([1 if a > 0.6072 else 0 for a in clf.predict(data[1])], data[3]))
    pdb.set_trace()
    conf_mat = confusion_matrix(data[3], y_pred)
    print("Lasso")
    print(conf_mat)
    fpr,tpr,_ = roc_curve(data[3], clf.predict(data[1]))
    
    temp_data = data[0][['Is.Non.Annual.', 'SingleGradeTripFlag', 'SPR.New.Existing', 'NPS 2010']]
    temp_data1 = data[1][['Is.Non.Annual.', 'SingleGradeTripFlag', 'SPR.New.Existing', 'NPS 2010']]
    clf = LogisticRegression(random_state=0, max_iter=10000).fit(temp_data, data[2])
    # pdb.set_trace()
    # print("Train Accuracy:")
    # print(clf.score(temp_data, data[2]))
    # print("Test Accuracy:")
    # print(accuracy_score(clf.predict(temp_data1), data[3]))
    # pdb.set_trace()
    yprobs2 = clf.predict_proba(temp_data1).T[1]
    y_pred = [1 if a > 0.6072 else 0 for a in yprobs2]
    conf_mat = confusion_matrix(data[3], y_pred)
    print("LogRegr (fwd stepwise)")
    print(conf_mat)
    fpr2,tpr2,_ = roc_curve(data[3], yprobs2)
    
    plt.plot(fpr,tpr,c='b', label='Lasso')
    plt.plot(fpr2,tpr2,c='r', label='LogRegr')
    plt.xlabel("False Pos")
    plt.xlabel("True Pos")
    plt.legend()
    plt.savefig("roc_curve_logr.png", dpi=300)
    plt.close()
    
    print("Lasso AUC:  {}".format(str(sum(tpr)/len(fpr))))
    print("LogRegr AUC:  {}".format(str(sum(tpr2)/(len(fpr2)+1))))
    
    
    percentages1, gains1 = cumulative_gain_curve(data[3], yprobs1, 1)
    percentages2, gains2 = cumulative_gain_curve(data[3], yprobs2, 1)
    gains1 = gains1 / percentages1
    gains2 = gains2 / percentages2
    plt.plot(percentages1, gains1, c="b", label="Lasso")
    plt.plot(percentages2, gains2, c="r", label="LogRegr")
    plt.plot(np.linspace(0,1,100), [1]*100, '--', c='k')
    plt.legend()
    plt.savefig("lift_curve_logr.png", dpi=300)
    plt.close()
    
    profs1 = calcProf(data[3], yprobs1, 60, 40)
    profs2 = calcProf(data[3], yprobs2, 60, 40)
    plt.plot(profs1.keys(), profs1.values(), c="b", label="Lasso")
    plt.plot(profs2.keys(), profs2.values(), c="r", label="LogRegr")
    plt.plot(np.linspace(0,1,100), [1]*100, '--', c='k')
    plt.legend()
    plt.savefig("profit.png", dpi=300)
    plt.close()
    
    import operator
    print("Lasso - Max Prof:  {}   Groups Contacted:  {}".format(max(profs1.items(), key=operator.itemgetter(1))[1], max(profs1.items(), key=operator.itemgetter(1))[0]))
    print("LogRegr - Max Prof:  {}   Groups Contacted:  {}".format(max(profs2.items(), key=operator.itemgetter(1))[1], max(profs2.items(), key=operator.itemgetter(1))[0]))
    
    
def calcProf(ytrue, yprob, gain, loss):
    # data = pd.DataFrame(data=[ytrue.vallues, list(yprob)], columns=['ytrue', 'ypred'])
    data = pd.DataFrame()
    data['ytrue'] = ytrue.values
    data['yprob'] = yprob
    data = data.sort_values(by=['yprob'], ascending=False)
    profit = { 0: 0}
    ct = 1
    for ix, vals in data.iterrows():
        if vals['ytrue']:
            profit[ct] = profit[ct-1] + gain
        else:
            profit[ct] = profit[ct-1] - loss
        ct += 1
    return profit
    

def calculate_bic(data, y, clf):
    bic = len(data) * np.log(mean_squared_error(clf.predict(data), y)) + len(data.columns) * np.log(len(data))
    return bic


def q6models(data):
    # features = ['SingleGradeTripFlag', 'Is.Non.Annual.', 'SPR.New.Existing', 'Total.Pax', 'Latest.RPL', 
    #             'Total.School.Enrollment', 'FPP.to.PAX', 'Initial.System.Date', 
    #             'FRP.Take.up.percent.', 'Departure.Date', 'SPR.Group.Revenue']
                
    features = ['NPS 2010', 'SingleGradeTripFlag', 'Is.Non.Annual.', 'Initial.System.Date', 
                'Total.Pax', 'FPP.to.School.enrollment', 'Total.School.Enrollment', 
                'Latest.RPL', '>= 3 FPP Date', 'DifferenceTraveltoFirstMeeting', 'SPR.New.Existing', 
                'FPP.to.PAX', '>= 10 FPP Date']
    data[0] = data[0][features]
    data[1] = data[1][features]
    
    tree = DecisionTreeClassifier(random_state=6, max_leaf_nodes=50, max_depth=10)
    tree.fit(data[0], data[2])
    tree_probs = tree.predict_proba(data[1]).T[1]
    tree_pred = [1 if a > 0.6072 else 0 for a in tree_probs]
    print("DecTree Test Accuracy:")
    print(accuracy_score([1 if a > 0.6072 else 0 for a in tree.predict(data[1])], data[3]))
    boost =  GradientBoostingRegressor(n_estimators=100, max_leaf_nodes=100, max_depth=20, random_state=6)
    boost.fit(data[0], data[2])
    boost_probs = boost.predict(data[1])
    boost_pred = [1 if a > 0.6072 else 0 for a in boost_probs]
    # print("Train Accuracy:")
    # print(accuracy_score([1 if a > 0.6072 else 0 for a in boost.predict(data[0])], data[2]))
    print("Boost Test Accuracy:")
    print(accuracy_score([1 if a > 0.6072 else 0 for a in boost.predict(data[1])], data[3]))
    rfc = RandomForestRegressor(n_estimators=100, random_state=1, max_leaf_nodes=100, max_depth=20, n_jobs=1)
    rfc.fit(data[0], data[2])
    rfc_probs = rfc.predict(data[1])
    rfc_pred = [1 if a > 0.6072 else 0 for a in rfc_probs]
    print("RFC Test Accuracy:")
    print(accuracy_score([1 if a > 0.6072 else 0 for a in rfc.predict(data[1])], data[3]))
    
    # feats = boost.feature_importances_
    # feats = dict(sorted(zip(data[0].columns, feats), key=lambda item: -item[1]))
    # for k, v in feats.items():
    #     print(k + ":  " + str(v))
    
    # ROC
    fpr,tpr,_ = roc_curve(data[3], tree_probs)
    fpr2,tpr2,_ = roc_curve(data[3], boost_probs)
    fpr3,tpr3,_ = roc_curve(data[3], rfc_probs)
    plt.plot(fpr,tpr,c='b', label='DecTree')
    plt.plot(fpr2,tpr2,c='r', label='Boost')
    plt.plot(fpr3,tpr3,c='k', label='RFC')
    plt.xlabel("False Pos")
    plt.xlabel("True Pos")
    plt.legend()
    plt.savefig("roc_curve_models.png", dpi=300)
    plt.close()
    
    print("DecTree AUC:  {}".format(str(sum(tpr)/len(fpr))))
    print("Boost AUC:  {}".format(str(sum(tpr2)/len(fpr2))))
    print("RFC AUC:  {}".format(str(sum(tpr3)/len(fpr3))))
    
    percentages1, gains1 = cumulative_gain_curve(data[3], tree_probs, 1)
    gains1 = gains1[14:]
    percentages1 = percentages1[14:]
    percentages2, gains2 = cumulative_gain_curve(data[3], boost_probs, 1)
    gains2 = gains2[14:]
    percentages2 = percentages2[14:]
    percentages3, gains3 = cumulative_gain_curve(data[3], rfc_probs, 1)
    gains1 = gains1 / percentages1
    gains2 = gains2 / percentages2
    gains3 = gains3 / percentages3
    plt.plot(percentages1, gains1, c="b", label="DecTree")
    plt.plot(percentages2, gains2, c="r", label="Boost")
    plt.plot(percentages3, gains3, c="k", label="RFC")
    plt.plot(np.linspace(0,1,100), [1]*100, '--', c='k')
    plt.legend()
    plt.savefig("lift_curve_models.png", dpi=300)
    plt.close()
    
    profs1 = calcProf(data[3], tree_probs, 60, 40)
    profs2 = calcProf(data[3], boost_probs, 60, 40)
    profs3 = calcProf(data[3], rfc_probs, 60, 40)
    plt.plot(profs1.keys(), profs1.values(), c="b", label="DecTree")
    plt.plot(profs2.keys(), profs2.values(), c="r", label="Boost")
    plt.plot(profs3.keys(), profs3.values(), c="k", label="RFC")
    plt.plot(np.linspace(0,1,100), [1]*100, '--', c='k')
    plt.legend()
    plt.savefig("profit_models.png", dpi=300)
    plt.close()
    
    import operator
    print("DecTree - Max Prof:  {}   Groups Contacted:  {}".format(max(profs1.items(), key=operator.itemgetter(1))[1], max(profs1.items(), key=operator.itemgetter(1))[0]))
    print("Boost - Max Prof:  {}   Groups Contacted:  {}".format(max(profs2.items(), key=operator.itemgetter(1))[1], max(profs2.items(), key=operator.itemgetter(1))[0]))
    print("RFC - Max Prof:  {}   Groups Contacted:  {}".format(max(profs3.items(), key=operator.itemgetter(1))[1], max(profs2.items(), key=operator.itemgetter(1))[0]))


if __name__ == '__main__':
    use_correct = True
    if use_correct:
        # data = get_correct_data()
        data = get_correct_combined()
    else:
        data = get_data()
        data = cleandata(data)
    MASTER = data
    data = factorize(data)
    train_test = traintest(data)
    # q5(train_test)
    # q5lasso(train_test)
    # q6(train_test)
    q6models(train_test)
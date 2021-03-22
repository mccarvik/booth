import pdb
import csv
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as sm
from scikitplot.helpers import cumulative_gain_curve
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.linear_model import Lasso
import scipy.stats as scipy
from sklearn.svm import SVC
from sklearn.utils import resample


def getstuff2(filename):
    data = []
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            if len(data) < 10000:
                data.append(row)
            else:
                yield pd.DataFrame(data)
                data = []
    return "end", data


def get_data():
    data = []
    data.append(pd.read_csv("trainx.csv").drop(columns=['Unnamed: 0']))
    data.append(pd.read_csv("trainy.csv").drop(columns=['Unnamed: 0']))
    data.append(pd.read_csv("testx.csv").drop(columns=['Unnamed: 0']))
    data.append(pd.read_csv("testy.csv").drop(columns=['Unnamed: 0']))
    return data
    
    data = pd.DataFrame()
    while True:
        for chunk in getstuff2("eureka.csv"):
            if len(chunk)==2: 
                data = pd.concat(data, chunk[1])
                print(len(data))
                brk = True
                break
            else:
                if data.empty:
                    data = chunk
                    new_header = data.iloc[0] #grab the first row for the header
                    data = data[1:] #take the data less the header row
                    data.columns = new_header #set the header row as the df header
                else:
                    chunk.columns = data.columns
                    data = pd.concat([data, chunk])
                print(len(data))
            
            if len(data) > 100000:
                brk = True
                break
            #     pdb.set_trace()
            #     data = pd.DataFrame()
        if brk:
            break
    # data = pd.read_csv("eureka.csv", low_memory=False)
    # data = pd.read_csv("eureka.csv", skiprows = lambda x: False if x > 10 else True)
    data = fixna(data)
    traintest(data)


def traintest(data):
    trainx, testx, trainy, testy = train_test_split(data.drop(columns=['converted_in_7days']), data['converted_in_7days'], test_size=0.50, random_state=5)
    trainx.to_csv("trainx.csv")
    trainy.to_csv("trainy.csv")
    testx.to_csv("testx.csv")
    testy.to_csv("testy.csv")


def fixna(data):
    """
    """
    strs = ['region', 'sourceMedium', 'device']
    factor_cat = []
    for col in data.columns:
        if len(data[col].unique()) == 2 or col in strs:
            factor_cat.append(col)
    
    date_cat = ['date']
    for cat in factor_cat:
        data[cat] = pd.factorize(data[cat])[0]
    
    for cat in date_cat:
        data[cat] = pd.to_datetime(data[cat])
    
    for col in data.columns:
        if data[data[col] == ''].empty:
            print("{} is fine".format(col))
            continue
        
        if col not in factor_cat and col not in date_cat:
            data[col] = data[col].replace(r'^\s*$', np.nan, regex=True)
            data[col] = data[col].fillna("0")
            data[col] = pd.to_numeric(data[col])
            data[col] = preprocessing.scale(data[col])
        elif col in date_cat:
            data[col] = data[col].replace(r'^\s*$', np.nan, regex=True)
            data[col] = data[col].fillna(dt.datetime(1900,1,1,0,0))
        else:
            data[col] = data[col].replace(r'^\s*$', np.nan, regex=True)
            data[col] = data[col].fillna("FIXED_NA")
        print("{} edited".format(col))
    
    data['converted_in_7days'] = [1 if float(x) > 0 else 0 for x in data['converted_in_7days']]
    
    for col in date_cat:
        data[cat] = pd.to_datetime(data[cat]).astype('int64')
        max_a = data[cat].max()
        min_a = data[cat].min()
        min_norm = -1
        max_norm =1
        data[cat] = (data[cat] - min_a) *(max_norm - min_norm) / (max_a-min_a) + min_norm
    
    
    return data


def sample(data, up=False):
    # Separate majority and minority classes
    train = data[0].merge(data[1], right_index=True, left_index=True)
    df_majority = train[train.converted_in_7days==0]
    df_minority = train[train.converted_in_7days==1]
    
    if not up: 
        # down sample majority class
        df_majority_upsampled = resample(df_majority, 
                                         replace=False,     # sample with replacement
                                         n_samples=len(df_minority),    # to match majority class
                                         random_state=123) # reproducible results
        # Combine majority class with upsampled minority class
        df_sampled = pd.concat([df_majority_upsampled, df_minority])
    else:
        # Upsample minority class
        df_minority_upsampled = resample(df_minority, 
                                         replace=True,     # sample with replacement
                                         n_samples=len(df_majority),    # to match majority class
                                         random_state=123) # reproducible results
        # Combine majority class with upsampled minority class
        df_sampled = pd.concat([df_majority, df_minority_upsampled])
    # Display new class counts
    print(df_sampled.converted_in_7days.value_counts())
    return df_sampled.drop(columns=['converted_in_7days']), df_sampled['converted_in_7days']
    

def q3(data):
    """
    """
    feats = ['DemoReqPg_CallClicks_evt_count', 'bounces', 'bounces_hist', 
            'contactus_top',
           'date', 'demo_page_top',
           'dsls', 'fired_DemoReqPg_CallClicks_evt',
           'goal4Completions',
           'offer_page_top', 'pageviews',
           'pageviews_hist', 'paid', 'paid_hist',
           'region',
           'sessionDuration', 'sessionDuration_hist', 'sessions', 'sessions_hist',
           'sourceMedium',
           'visited_demo_page',
           'visited_demo_page_hist',
           'visited_vacuum_cleaner_page_hist',
           'visited_water_purifier_page_hist',
           'water_purifier_page_top']
    # data[0] = data[0][feats]
    # data[2] = data[2][feats]
    
    data[0] = data[0][:20000]
    data[1] = data[1][:20000]
    data[0], data[1] = sample([data[0], data[1]], True)
    
    
    
    print("start")
    # svc = SVC(probability=True)
    # svc.fit(data[0], data[1])
    # svc_pred = svc.predict(data[2])
    # svc_proba = svc.predict_proba(data[2]).T[1]
    # print("SVC Test Accuracy:")
    # print(accuracy_score(svc_pred, data[3]))
    
    boost = GradientBoostingClassifier(n_estimators=18, random_state=1, max_leaf_nodes=200, max_depth=20)
    boost.fit(data[0], data[1])
    boost_pred = boost.predict(data[2])
    boost_proba = boost.predict_proba(data[2]).T[1]
    print("SVC Test Accuracy:")
    print(accuracy_score(boost_pred, data[3]))
    
    logr =  LogisticRegression()
    logr.fit(data[0], data[1])
    logr_pred = logr.predict(data[2])
    logr_proba = logr.predict_proba(data[2]).T[1]
    # logr_pred = [1 if a > 0.50 else 0 for a in logr_probs]
    print("Logr Test Accuracy:")
    print(accuracy_score(logr_pred, data[3]))
    
    rfc = RandomForestClassifier(n_estimators=100, random_state=1, max_leaf_nodes=200, 
                                 max_depth=20, n_jobs=1)
    rfc.fit(data[0], data[1])
    rfc_pred = rfc.predict(data[2])
    rfc_proba = rfc.predict_proba(data[2]).T[1]
    # rfc_pred = [1 if a > 0.6072 else 0 for a in rfc_probs]
    print("RFC Test Accuracy:")
    print(accuracy_score(rfc_pred, data[3]))
    
    # feats = rfc.feature_importances_
    # feats = dict(sorted(zip(data[0].columns, feats), key=lambda item: -item[1]))
    # for k, v in feats.items():
    #     print(k + ":  " + str(v))
    

    # ROC
    fpr,tpr,_ = roc_curve(data[3], boost_proba)
    fpr2,tpr2,_ = roc_curve(data[3], logr_proba)
    fpr3,tpr3,_ = roc_curve(data[3], rfc_proba)
    plt.plot(fpr,tpr,c='b', label='SVM')
    plt.plot(fpr2,tpr2,c='r', label='Logr')
    plt.plot(fpr3,tpr3,c='k', label='RFC')
    plt.xlabel("False Pos")
    plt.xlabel("True Pos")
    plt.legend()
    plt.savefig("roc_curve_models_down.png", dpi=300)
    plt.close()
    
    print("SVM AUC:  {}".format(str(sum(tpr)/len(fpr))))
    print("Logr AUC:  {}".format(str(sum(tpr2)/len(fpr2))))
    print("RFC AUC:  {}".format(str(sum(tpr3)/len(fpr3))))
    
    percentages1, gains1 = cumulative_gain_curve(data[3], boost_proba, 1)
    gains1 = gains1[50:]
    percentages1 = percentages1[50:]
    percentages2, gains2 = cumulative_gain_curve(data[3], logr_proba, 1)
    gains2 = gains2[50:]
    percentages2 = percentages2[50:]
    percentages3, gains3 = cumulative_gain_curve(data[3], rfc_proba, 1)
    gains3 = gains3[50:]
    percentages3 = percentages3[50:]
    gains1 = gains1 / percentages1
    gains2 = gains2 / percentages2
    gains3 = gains3 / percentages3
    plt.plot(percentages1, gains1, c="b", label="SVC")
    plt.plot(percentages2, gains2, c="r", label="Logr")
    plt.plot(percentages3, gains3, c="k", label="RFC")
    plt.plot(np.linspace(0,1,100), [1]*100, '--', c='k')
    plt.legend()
    plt.savefig("lift_curve_models_down.png", dpi=300)
    plt.close()


if __name__ == '__main__':
    data = get_data()
    q3(data)
    
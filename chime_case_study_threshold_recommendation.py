# Import Required Packages
# !conda install -y -c conda-forge xgboost==0.90
# !conda install -y -c conda-forge shap
# !pip install imbalanced-learn

import os, boto3, re, json, pickle, gzip, urllib.request, ast, struct, io, gc
import sagemaker
from sagemaker import get_execution_role
from sagemaker.xgboost.model import XGBoostModel
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
import xgboost as xgb
import sklearn as sk
import pandas as pd
import numpy as np
from numpy import argmax
import matplotlib.pyplot as plt
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, confusion_matrix, average_precision_score, precision_recall_curve
import shap
from IPython.core.display import display, HTML
import warnings
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE 
import tarfile, zipfile

# import the dataset 
members = pd.read_csv('member_disputes_cleaned.csv')
members.columns
members.head()

# plot the distribution for RISK SCORE
sns.displot(
  data=members,
  x="RISK_SCORE", 
    col="TYPE",
  kind="hist",
  aspect=1.4
)
# look at the quantiles on RISK SCORE by TYPE to understand the skewness of distribution
def q1(x):
    return x.quantile(0.25)

def q3(x):
    return x.quantile(0.75)

f = {'RISK_SCORE': ['mean','median', 'std', q1, q3]}
df1 = members.groupby('TYPE').agg(f)
df1
# create a list of thresholds
import decimal
def float_range(start, stop, step):
      while start < stop:
        yield float(start)
        start += decimal.Decimal(step)

threshold = list(float_range(0, 1, '0.01'))

# create a function returning the new profit margin for each threshold 
def flag_dataset(df, val):
    df['fraud_flag'] = df.apply(lambda x : 1 if x['RISK_SCORE'] > val else 0 , axis = 1)
    df['NET_PROFIT_updated'] = df.apply(lambda x : -x['NET_PROFIT'] if x['RISK_SCORE'] > val else x['NET_PROFIT'], axis = 1)
    df_group = df.groupby(by=['TYPE','fraud_flag'],  as_index=False).agg(net_profit = ('NET_PROFIT_updated', 'sum'), member_cnt = ('USER_ID','count') )
    df_group['threshold'] = val
    return df_group 

# loop over a list of threshold and return the business metrics for each score 
dfs = pd.DataFrame(columns=('TYPE','fraud_flag','net_profit','member_cnt', 'threshold'))
for thr in threshold:
    df_group = flag_dataset(members_50, thr)
    dfs = dfs.append(df_group)

members_loss = dfs.loc[dfs['fraud_flag'] == 1,['TYPE','member_cnt', 'threshold'] ]
members_loss = members_loss.rename(columns={"member_cnt": "member_loss_cnt"})
members_keep = dfs.loc[dfs['fraud_flag'] == 0,['TYPE','member_cnt', 'threshold'] ]
members_keep = members_keep.rename(columns={"member_cnt": "member_cnt"})

dfs_cleaned = dfs.groupby(by=['TYPE','threshold'], as_index=False).agg(net_profit= ('net_profit', 'sum'))

# self join twice to get the number of members we might keep and the number of user we may lose given its threshold
dfs_cleaned_grouped = dfs_cleaned.merge(members_loss, how='left', on=['TYPE','threshold']).merge(members_keep, how='left', on=['TYPE','threshold'])

# calculate the average net profit margin 
dfs_cleaned_grouped['avg_net_profit_margin'] = dfs_cleaned_grouped['net_profit'] / dfs_cleaned_grouped['member_cnt']

# plot threshold v.s cost v.s revenue 
def plot_threshold_vs_cost(net_profit, member_loss_cnt, threshold):
    """
    Modified from:
    Hands-On Machine learning with Scikit-Learn
    and TensorFlow; p.89
    """
    plt.figure(figsize=(15, 15))
    plt.title("decision threshold")
    plt.plot(threshold, net_profit, "b--", label="Average Net Profit Margin")
    plt.plot(threshold, member_loss_cnt, "g-", label="Member Loss")
    plt.ylabel("Score")
    plt.xlabel("Decision Threshold")
    plt.legend(loc='best')

 plot_threshold_vs_cost(dfs_cleaned_grouped.loc[dfs_cleaned_grouped['TYPE'] == 'NON-DD','net_profit'],
                       dfs_cleaned_grouped.loc[dfs_cleaned_grouped['TYPE'] == 'NON-DD','member_loss_cnt'],
                       dfs_cleaned_grouped.loc[dfs_cleaned_grouped['TYPE'] == 'NON-DD','threshold']
                      )
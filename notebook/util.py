import time
import csv

import numpy as np
import pandas as pd
import urllib, json
from simpledbf import Dbf5

import matplotlib as mpl
import matplotlib.pylab as plt
import dask.dataframe as dd
import seaborn as sns
sns.set()
sns.set_color_codes()

from scipy.stats.stats import pearsonr
import scipy.stats as sp
import statsmodels.api as sm

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def repr_all(df):
    len_df = len(df)
    pd.set_option('display.max_rows', len_df)
    display(df)
    pd.reset_option('display.max_rows')
    return '{} rows printed'.format(len_df)


# get labem from lda model
def get_labels(model, tres, sentence):
    lda = model
    proba_list = lda[sentence]
    res = [x for x in proba_list if x[1] >= tres]
    if len(res) == 0:
        res = None
    return res
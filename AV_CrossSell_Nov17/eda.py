# EDA
# split sample into train and test
# feature calculation
# preprocessing
# feature selection
# model development
# model validation

import pandas as pd
import numpy as np
import joblib
import json
import os

# GLOBALS
local_data_root = '/Users/vnathan/Documents/analyticsvidhya/cross_sell/'
train_path = local_data_root + 'train.csv'
test_path = local_data_root + 'test_plBmD8c.csv'
DV = 'RESPONDERS'
dv_map = {'Y': 1, 'N': 0}


def read_datafile(path):
    fn, ext = os.path.splitext(path)
    read_fn = {'.csv': pd.read_csv,
               '.pkl': joblib.load,
               '.json': json.load}
    return read_fn.get(ext, pd.read_csv)(path)


def desc_categorical(x, y):
    """
    describes x where x and y are categorical
    """
    print 'value_counts'
    print x.value_counts()
    table = pd.crosstab(x, y)
    table['event_rate'] = map(lambda a, b: a/float(a+b), table['Y'],
                              table['N'])
    print table


def desc_numeric(x, y):
    print 'correlation:'
    mask = (x.notnull()) & (y.notnull())
    print np.corrcoef(x[mask], y[mask])[0, 1]


if __name__ == '__main__':
    print 'EDA...'
    # read data
    train = read_datafile(train_path)
    test = read_datafile(test_path)
    # DV
    train[DV] = train[DV].map(dv_map)
    # insights
    types = train.dtypes
    obj_dtype = types[types == object].index.tolist()
    num_dtype = types[types != object].index.tolist()
    num_dtype.remove('CUSTOMER_ID')

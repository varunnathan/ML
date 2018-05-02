import pandas as pd
import numpy as np
import joblib
import json
import os
import pickle
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split
from math import log
from utility import MaxAUCImputerCV, LegacyOutlierScaler, cluster_features
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier  # GBM algorithm
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

# GLOBALS
local_data_root = '/Users/vnathan/Documents/analyticsvidhya/cross_sell/'
train_path = local_data_root + 'train.csv'
test_path = local_data_root + 'test_plBmD8c.csv'
interim_path = local_data_root + 'intermediate/'
prepared_data_file = 'prepared_data.csv'
preprocess_out_file = 'train_data_engineered_preprocessed.pkl'
featsel_out_file = 'train_data_engineered_preprocessed_{}.pkl'
featsel_out_test_file = 'test_data_engineered_preprocessed_{}.pkl'
DV = 'RESPONDERS'
sample_col = 'sample'
dv_map = {'Y': 1, 'N': 0}
cols_remove = ['CUSTOMER_ID', 'OCCUP_ALL_NEW']
date_cols = ['MATURITY_GL', 'MATURITY_LAP', 'MATURITY_LAS', 'CLOSED_DATE']
flag_cols = [x+'_flag' for x in date_cols]
maturity_cols = ['closed_before_maturity_{}'.format(x) for x in [
 'GL', 'LAP', 'LAS']]
time_since_closed_cols = ['time_since_closed_{}'.format(x) for x in [
 'GL', 'LAP', 'LAS']]
feat_groups = ('CSF')
PREPROCESS = {
    'exoutscaler': LegacyOutlierScaler(),
    'exaucimputer': MaxAUCImputerCV(),
    'stdscaler': StandardScaler()
}
steps = ['exaucimputer', 'exoutscaler', 'stdscaler']
SIMPLE = {'min': np.nanmin, 'max': np.nanmax, 'sum': np.nansum,
          'average': np.nanmean, 'perc50': np.nanmedian, 'std': np.nanstd}


def convert_to_date(x, format='%d%b%Y'):
    return pd.to_datetime(x, format=format)


def get_feature_cols(data):
    df = data.copy()
    cols = [x for x in list(df.columns) if (x not in cols_remove) & (x != DV)]
    return cols


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
    table['event_rate'] = map(lambda a, b: a/float(a+b), table[1],
                              table[0])
    print table


def desc_numeric(x, y):
    print 'correlation:'
    mask = (x.notnull()) & (y.notnull())
    print np.corrcoef(x[mask], y[mask])[0, 1]


def get_woe(x, y):
    table = pd.crosstab(x, y)
    table.reset_index(inplace=True)
    table.rename(columns={0: '#good', 1: '#bad'}, inplace=True)
    total_good = table['#good'].sum()
    total_bad = table['#bad'].sum()
    table['perc_good'] = table['#good'].apply(lambda x: x/float(total_good))
    table['perc_bad'] = table['#bad'].apply(lambda x: x/float(total_bad))
    mask = (table['perc_good'] != 0) & (table['perc_bad'] != 0)
    table.loc[mask, 'WOE'] = map(
     lambda x, y: log(x / float(y)), table.loc[mask, 'perc_good'],
     table.loc[mask, 'perc_bad'])
    table.loc[~mask, 'WOE'] = np.NaN
    table.reset_index(drop=True, inplace=True)
    return table


def get_iv(x, y):
    table = get_woe(x, y)
    mask = table['WOE'].notnull()
    table.loc[mask, 'IV'] = (
     table.loc[mask, 'perc_good'] - table.loc[mask, 'perc_bad'])*table.loc[
      mask, 'WOE']
    table.loc[~mask, 'IV'] = np.NaN
    return np.nansum(table['IV'])


def num_missing(x):
    """Returns number of missing values of a series"""
    return np.sum(pd.isnull(x))


def num_unique(x):
    """Returns number of unique values of a series"""
    return x.nunique()


def get_perc5(x):
    """Returns the 5th percentile of a series"""
    try:
        return np.percentile(x[pd.notnull(x)], 5)
    except:
        return None


def get_perc25(x):
    """Returns the 25th percentile of a series"""
    try:
        return np.percentile(x[pd.notnull(x)], 25)
    except:
        return None


def get_perc75(x):
    """Returns the 25th percentile of a series"""
    try:
        return np.percentile(x[pd.notnull(x)], 75)
    except:
        return None


def get_perc95(x):
    """Returns the 95th percentile of a series"""
    try:
        return np.percentile(x[pd.notnull(x)], 95)
    except:
        return None


def add_maturity_cols(data):
    df = data.copy()
    for i, col in enumerate(date_cols):
        print col
        print 'flag columns:'
        mask = df[col].notnull()
        df.loc[mask, flag_cols[i]] = 1
    for i, col in enumerate(date_cols[:3]):
        print col
        print 'maturity columns:'
        mask = (df['CLOSED_DATE'].notnull()) & (df[col].notnull())
        mask1 = df['CLOSED_DATE'] <= df[col]
        df.loc[(mask & mask1), maturity_cols[i]] = 1
        print 'time since closed columns:'
        df.loc[mask, time_since_closed_cols[i]] = map(
         lambda x, y: abs((x-y).days), df.loc[mask, col],
         df.loc[mask, 'CLOSED_DATE'])
    return df


def impute_nulls(data, feat_cols):
    df = data.copy()
    for col in feat_cols:
        print col
        mask = df[col].isnull()
        if df[col].dtypes == object:
            df.loc[mask, col] = 'N'
            df[col] = df[col].map(dv_map)
        else:
            df.loc[mask, col] = 0
            df[col] = df[col].astype(int)
    return df


def get_summary_stat_numeric(data, feat_cols, sample=None, out=None):

    if sample:
        dev = data[data[sample_col] == sample]
    else:
        dev = data.copy()

    summary1 = pd.DataFrame(
     {'SNo': range(1, len(feat_cols)+1), 'Feature': feat_cols})
    summary1['dtype'] = pd.Series(dev[feat_cols].dtypes.values)
    summary1['min'] = pd.Series(dev[feat_cols].apply(np.min, axis=0).values)
    summary1['max'] = pd.Series(dev[feat_cols].apply(np.max, axis=0).values)
    summary1['mean'] = pd.Series(dev[feat_cols].apply(
     np.nanmean, axis=0).values)
    summary1['median'] = pd.Series(dev[feat_cols].apply(
     np.nanmedian, axis=0).values)
    summary1['misscnt'] = pd.Series(dev[feat_cols].apply(
     num_missing, axis=0).values)
    summary1['misspct'] = summary1['misscnt'].apply(
     lambda x: x/float(len(dev))*100)
    summary2 = summary1.copy()
    summary2['std'] = pd.Series(dev[feat_cols].apply(np.std, axis=0).values)
    summary2['Perc5'] = pd.Series(
     dev[feat_cols].apply(get_perc5, axis=0).values)
    summary2['Perc25'] = pd.Series(
     dev[feat_cols].apply(get_perc25, axis=0).values)
    summary2['Perc75'] = pd.Series(
     dev[feat_cols].apply(get_perc75, axis=0).values)
    summary2['Perc95'] = pd.Series(
     dev[feat_cols].apply(get_perc95, axis=0).values)
    summary2['uniquecnt'] = pd.Series(
     dev[feat_cols].apply(num_unique, axis=0).values)
    summary2['uniquepct'] = summary2['uniquecnt'].apply(
     lambda x: x/float(len(dev))*100)
    for i in range(len(summary2)):
        print summary2.loc[i, 'Feature']
        summary2.loc[i, 'auc'] = roc_auc_score(
         dev[DV], dev[summary2.loc[i, 'Feature']])
    if out:
        summary2.to_csv(out, index=False)
    else:
        return summary2


def IV_categorical(data, feat_cols, sample=None, out=None):

    if sample:
        dev = data[data[sample_col] == sample]
    else:
        dev = data.copy()

    summary = pd.DataFrame(
     {'SNo': range(1, len(feat_cols)+1), 'Feature': feat_cols})
    summary['uniquecnt'] = pd.Series(
     dev[feat_cols].apply(num_unique, axis=0).values)
    for i in range(len(summary)):
        print summary.loc[i, 'Feature']
        if summary.loc[i, 'uniquecnt'] > 1:
            summary.loc[i, 'IV'] = get_iv(
             dev[summary.loc[i, 'Feature']], dev[DV])

    if out:
        summary.to_csv(out, index=False)
    else:
        return summary


def create_addl_features(data):
    df = data.copy()
    df['CSF_N_CASA_MAX_DIFF_MIN_BALANCE_MTD'] = df[
     'CSF_N_CASA_MAX_BALANCE_MTD'] - df['CSF_N_CASA_MIN_BALANCE_MTD']
    mask = (df['CSF_N_CASA_MIN_BALANCE_MTD'] != 0)
    df.loc[mask, 'CSF_N_CASA_MAX_RATIO_MIN_BALANCE_MTD'] = map(
     lambda x, y: 1.*x/y, df.loc[mask, 'CSF_N_CASA_MAX_BALANCE_MTD'],
     df.loc[mask, 'CSF_N_CASA_MIN_BALANCE_MTD'])

    print 'ratio of spends'
    dc_spend = ['CSF_DC_SPEND_MON_0{}'.format(i) for i in range(1, 7)]
    cc_spend = ['CSF_CC_SPEND_MON_0{}'.format(i) for i in range(1, 7)]
    total_spend = cc_spend + dc_spend
    for i, col in enumerate(cc_spend):
        print dc_spend[i], col
        mask = (df[dc_spend[i]].notnull()) & (df[col].notnull()) & (
         df[dc_spend[i]] != 0)
        df.loc[mask, col+'_RATIO_DC_SPEND'] = map(
         lambda x, y: 1.*x/y, df.loc[mask, col], df.loc[mask, dc_spend[i]])

    print 'acceleration in spends'
    for col in cc_spend[1:]:
        mask = (df[col].notnull()) & (df[cc_spend[0]].notnull()) & (
         df[col] != 0)
        name = cc_spend[0]+'_RATIO_'+col
        print name
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, cc_spend[0]], df.loc[mask, col])

    for col in dc_spend[1:]:
        mask = (df[col].notnull()) & (df[dc_spend[0]].notnull()) & (
         df[col] != 0)
        name = dc_spend[0]+'_RATIO_'+col
        print name
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, dc_spend[0]], df.loc[mask, col])

    print 'simple stats - spends'
    for key, fn in SIMPLE.iteritems():
        print key
        df['CSF_cc_spend_'+key] = fn(df[cc_spend], axis=1)
        df['CSF_dc_spend_'+key] = fn(df[dc_spend], axis=1)
        df['CSF_total_spend_'+key] = fn(df[total_spend], axis=1)

    print 'ratio of stmts and cr_limit'
    cc_stmts = ['CSF_STMT_MON_0{}'.format(i) for i in range(1, 4)]
    for col in cc_stmts:
        print col
        mask = (df[col].notnull()) & (df['CSF_CR_LIMIT'].notnull()) & (
         df['CSF_CR_LIMIT'] != 0)
        name = col+'_RATIO_CR_LIMIT'
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, col], df.loc[mask, 'CSF_CR_LIMIT'])

    print 'simple stats - cc_stmts'
    for key, fn in SIMPLE.iteritems():
        print key
        df['CSF_cc_stmts_'+key] = fn(df[cc_stmts], axis=1)

    print 'monthly balance'
    mon_bal = ['CSF_AMB_MON_0{}'.format(i) for i in range(1, 5)]

    print 'simple stats - mon_bal'
    for key, fn in SIMPLE.iteritems():
        print key
        df['CSF_mon_bal_'+key] = fn(df[mon_bal], axis=1)

    print 'acceleration - mon_bal'
    for col in mon_bal[1:]:
        mask = (df[col].notnull()) & (df[mon_bal[0]].notnull()) & (
         df[col] != 0)
        name = mon_bal[0]+'_RATIO_'+col
        print name
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, mon_bal[0]], df.loc[mask, col])

    print 'ratio of max_bal/min_bal and eop_bal'
    for col in ['CSF_N_CASA_MAX_BALANCE_MTD', 'CSF_N_CASA_MIN_BALANCE_MTD']:
        ratio = col + '_RATIO_EOP_BAL_MON_01'
        mask = (df[col].notnull()) & (df['CSF_EOP_BAL_MON_01'].notnull()) & (
         df[col] != 0)
        df.loc[mask, ratio] = map(
         lambda x, y: 1.*x/y, df.loc[mask, 'CSF_EOP_BAL_MON_01'],
         df.loc[mask, col])

    print 'ratio of CSF_AMB_MON_01 and CSF_EOP_BAL_MON_01'
    mask = (df['CSF_EOP_BAL_MON_01'].notnull()) & (
     df['CSF_EOP_BAL_MON_01'] != 0) & (df['CSF_AMB_MON_01'].notnull())
    ratio = 'CSF_AMB_MON_01_RATIO_EOP_BAL_MON_01'
    df.loc[mask, ratio] = map(
     lambda x, y: 1.*x/y, df.loc[mask, 'CSF_AMB_MON_01'],
     df.loc[mask, 'CSF_EOP_BAL_MON_01'])
    print 'ratio of CSF_AMB_MON_02 and CSF_EOP_MON_02'
    mask = (df['CSF_EOP_MON_02'].notnull()) & (
     df['CSF_EOP_MON_02'] != 0) & (df['CSF_AMB_MON_02'].notnull())
    ratio = 'CSF_AMB_MON_02_RATIO_EOP_MON_02'
    df.loc[mask, ratio] = map(
     lambda x, y: 1.*x/y, df.loc[mask, 'CSF_AMB_MON_02'],
     df.loc[mask, 'CSF_EOP_MON_02'])
    print 'ratio of CSF_AMB_MON_03 and CSF_EOP_MON_03'
    mask = (df['CSF_EOP_MON_03'].notnull()) & (
     df['CSF_EOP_MON_03'] != 0) & (df['CSF_AMB_MON_03'].notnull())
    ratio = 'CSF_AMB_MON_03_RATIO_EOP_MON_03'
    df.loc[mask, ratio] = map(
     lambda x, y: 1.*x/y, df.loc[mask, 'CSF_AMB_MON_03'],
     df.loc[mask, 'CSF_EOP_MON_03'])

    print 'simple stats in investments'
    dmat_invest = ['CSF_CDMAT_MON_0{}'.format(i) for i in [1, 4]]
    mf_invest = ['CSF_MF_MON_0{}'.format(i) for i in [1, 4]]
    rd_invest = ['CSF_RD_MON_0{}'.format(i) for i in [1, 4]]
    fd_invest = ['CSF_FD_MON_0{}'.format(i) for i in [1, 4]]
    li_invest = ['CSF_LI_MON_0{}'.format(i) for i in [1, 4]]
    gi_invest = ['CSF_GI_MON_0{}'.format(i) for i in [1, 4]]
    invest = (dmat_invest + mf_invest + rd_invest + fd_invest + li_invest +
              gi_invest)
    invest_dict = {'mf': mf_invest, 'rd': rd_invest, 'fd': fd_invest,
                   'li': li_invest, 'gi': gi_invest, 'dmat': dmat_invest,
                   'all': invest}
    for key, fn in SIMPLE.iteritems():
        for name, value in invest_dict.iteritems():
            df['CSF_{}_invest_'.format(name)+key] = fn(df[value], axis=1)

    print 'number of txns'
    dc_txn = ['CSF_DC_TXN_MON_0{}'.format(i) for i in range(1, 7)]
    cc_txn = ['CSF_CC_TXN_MON_0{}'.format(i) for i in range(1, 7)]
    total_txn = cc_txn + dc_txn
    print 'ratio of spends'
    for i, col in enumerate(cc_txn):
        print dc_txn[i], col
        mask = (df[dc_txn[i]].notnull()) & (df[col].notnull()) & (
         df[dc_txn[i]] != 0)
        df.loc[mask, col+'_RATIO_DC_TXN'] = map(
         lambda x, y: 1.*x/y, df.loc[mask, col], df.loc[mask, dc_txn[i]])

    print 'acceleration in spends'
    for col in cc_txn[1:]:
        print col
        mask = (df[col].notnull()) & (df[cc_txn[0]].notnull()) & (df[col] != 0)
        name = cc_txn[0]+'_RATIO_'+col
        print name
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, cc_txn[0]], df.loc[mask, col])

    for col in dc_txn[1:]:
        print col
        mask = (df[col].notnull()) & (df[dc_txn[0]].notnull()) & (df[col] != 0)
        name = dc_txn[0]+'_RATIO_'+col
        print name
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, dc_txn[0]], df.loc[mask, col])

    print 'simple stats in spends'
    for key, fn in SIMPLE.iteritems():
        print key
        df['CSF_cc_txn_'+key] = fn(df[cc_txn], axis=1)
        df['CSF_dc_txn_'+key] = fn(df[dc_txn], axis=1)
        df['CSF_total_txn_'+key] = fn(df[total_txn], axis=1)

    print 'average spend'
    for cat in ['DC', 'CC']:
        for i in range(1, 7):
            name = 'CSF_{}_AVG_SPEND_MON_0{}'.format(cat, i)
            print name
            mask = ((df['CSF_{}_SPEND_MON_0{}'.format(cat, i)].notnull()) &
                    (df['CSF_{}_TXN_MON_0{}'.format(cat, i)].notnull()) &
                    (df['CSF_{}_TXN_MON_0{}'.format(cat, i)] != 0))
            df.loc[mask, name] = map(
             lambda x, y: 1.*x/y, df.loc[mask, 'CSF_{}_SPEND_MON_0{}'.format(
              cat, i)], df.loc[mask, 'CSF_{}_TXN_MON_0{}'.format(cat, i)])

    print 'card spend by category'
    print 'ratios'
    spend_cat = 'CSF_{}_SPEND_MON_0{}_{}'
    spend = 'CSF_{}_SPEND_MON_0{}'
    categories = {'CC': ['ENT', 'MED', 'HMD', 'HBY', 'CARE', 'TRL', 'RST',
                         'JER', 'HTL', 'ATM'],
                  'DC': ['ENT', 'MED', 'GRC', 'HMD', 'HBY', 'CARE', 'TRL',
                         'RST', 'JER', 'HTL', 'ATM']}
    print 'ratios'
    print 'cc_spend'
    for key, value in categories.iteritems():
        for i in range(1, 6):
            if (key == 'CC') & (i == 5):
                continue
            denominator = spend.format(key, i)
            print denominator
            for item in value:
                if (key == 'DC') & (i in [1, 2, 3, 5]) & (item == 'GRC'):
                    continue
                numerator = spend_cat.format(key, i, item)
                print numerator
                mask = (df[numerator].notnull()) & (
                 df[denominator].notnull()) & (df[denominator] != 0)
                name = numerator+'_RATIO'+denominator.strip('CSF_')
                df.loc[mask, name] = map(
                 lambda x, y: 1.*x/y, df.loc[mask, numerator],
                 df.loc[mask, denominator])

    all_spend_cat = []
    for key, value in categories.iteritems():
        for i in range(1, 6):
            if (key == 'CC') & (i == 5):
                continue
            for item in value:
                if (key == 'DC') & (i in [1, 2, 3, 5]) & (item == 'GRC'):
                    continue
                col = spend_cat.format(key, i, item)
                print col
                all_spend_cat.append(col)
    cc_spend_cat = [x for x in all_spend_cat if x.find('_CC_') != -1]
    dc_spend_cat = [x for x in all_spend_cat if x.find('_DC_') != -1]
    print cc_spend_cat

    print 'simple stats'
    for key, fn in SIMPLE.iteritems():
        name = 'CSF_cc_spend_cat_'+key
        print name
        df[name] = fn(df[cc_spend_cat], axis=1)
        name = 'CSF_dc_spend_cat_'+key
        print name
        df[name] = fn(df[dc_spend_cat], axis=1)
        name = 'CSF_all_spend_cat_'+key
        print name
        df[name] = fn(df[all_spend_cat], axis=1)

    print 'debit and credit amt'
    debit_txn_amt = 'CSF_D_AMT_L3_MON_0{}'
    credit_txn_amt = 'CSF_C_AMT_L3_MON_0{}'
    debit_txn_count = 'CSF_D_COUNT_L3_MON_0{}'
    credit_txn_count = 'CSF_C_COUNT_L3_MON_0{}'
    max_credit_amt = 'CSF_MAX_C_AMT_L3_MON_0{}'
    for i in range(1, 7):
        name = 'CSF_total_txn_amt_MON_0{}'.format(i)
        print name
        df[name] = np.nansum(
         df[[debit_txn_amt.format(i), credit_txn_amt.format(i)]], axis=1)
        name = 'CSF_total_txn_count_MON_0{}'.format(i)
        print name
        df[name] = np.nansum(
         df[[debit_txn_count.format(i), credit_txn_count.format(i)]], axis=1)
        name = 'CSF_cashflow_MON_0{}'.format(i)
        print name
        df[name] = df[credit_txn_amt.format(i)] - df[debit_txn_amt.format(i)]
        name = 'CSF_credit_ratio_debit_amt_MON_0{}'.format(i)
        print name
        numerator = credit_txn_amt.format(i)
        denominator = debit_txn_amt.format(i)
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])
        name = 'CSF_credit_ratio_debit_count_MON_0{}'.format(i)
        print name
        numerator = credit_txn_count.format(i)
        denominator = debit_txn_count.format(i)
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])
        name = 'CSF_avg_debit_amt_MON_0{}'.format(i)
        print name
        numerator = debit_txn_amt.format(i)
        denominator = debit_txn_count.format(i)
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])
        name = 'CSF_avg_credit_amt_MON_0{}'.format(i)
        print name
        numerator = credit_txn_amt.format(i)
        denominator = credit_txn_count.format(i)
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])
        name = 'CSF_avg_total_amt_MON_0{}'.format(i)
        print name
        numerator = 'CSF_total_txn_amt_MON_0{}'.format(i)
        denominator = 'CSF_total_txn_count_MON_0{}'.format(i)
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])
        name = 'CSF_max_ratio_total_credit_amt_MON_0{}'.format(i)
        print name
        numerator = max_credit_amt.format(i)
        denominator = credit_txn_amt.format(i)
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])

    print 'simple stats - debit and credit amt'
    debit_txn_amt_cols = [debit_txn_amt.format(i) for i in range(1, 7)]
    credit_txn_amt_cols = [credit_txn_amt.format(i) for i in range(1, 7)]
    debit_txn_count_cols = [debit_txn_count.format(i) for i in range(1, 7)]
    credit_txn_count_cols = [credit_txn_count.format(i) for i in range(1, 7)]
    for key, fn in SIMPLE.iteritems():
        name = 'CSF_debit_txn_amt_'+key
        print name
        df[name] = fn(df[debit_txn_amt_cols], axis=1)
        name = 'CSF_credit_txn_amt_'+key
        print name
        df[name] = fn(df[credit_txn_amt_cols], axis=1)
        name = 'CSF_debit_txn_count_'+key
        print name
        df[name] = fn(df[debit_txn_count_cols], axis=1)
        name = 'CSF_credit_txn_count_'+key
        print name
        df[name] = fn(df[credit_txn_count_cols], axis=1)

    print 'NEFT_CC_AMOUNT, NEFT_CC_TXN, NEFT_DC_AMOUNT, NEFT_DC_TXN'
    for cat in ['CC', 'DC']:
        numerator = 'CSF_NEFT_{}_AMOUNT'.format(cat)
        denominator = 'CSF_NEFT_{}_TXN'.format(cat)
        name = 'CSF_AVG_{}_NEFT_AMOUNT'.format(cat)
        print name
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])

    df['CSF_TOTAL_NEFT_AMOUNT'] = np.nansum(
     df[['CSF_NEFT_CC_AMOUNT', 'CSF_NEFT_DC_AMOUNT']], axis=1)
    df['CSF_TOTAL_NEFT_TXN'] = np.nansum(
     df[['CSF_NEFT_CC_TXN', 'CSF_NEFT_DC_TXN']], axis=1)

    print 'TPT_CC_AMOUNT, TPT_CC_TXN, TPT_DC_AMOUNT, TPT_DC_TXN'
    for cat in ['CC', 'DC']:
        numerator = 'CSF_TPT_{}_AMOUNT_MON_01'.format(cat)
        denominator = 'CSF_TPT_{}_TXN_MON_01'.format(cat)
        name = 'CSF_AVG_{}_TPT_AMOUNT_MON_01'.format(cat)
        print name
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])

    df['CSF_TOTAL_TPT_AMOUNT_MON_01'] = np.nansum(
     df[['CSF_TPT_CC_AMOUNT_MON_01', 'CSF_TPT_DC_AMOUNT_MON_01']], axis=1)
    df['CSF_TOTAL_TPT_TXN_MON_01'] = np.nansum(
     df[['CSF_TPT_CC_TXN_MON_01', 'CSF_TPT_DC_TXN_MON_01']], axis=1)

    print 'IMPS_CC_AMOUNT, IMPS_CC_TXN'
    for cat in ['CC']:
        numerator = 'CSF_IMPS_{}_AMOUNT_MON_01'.format(cat)
        denominator = 'CSF_IMPS_{}_TXN_MON_01'.format(cat)
        name = 'CSF_AVG_{}_IMPS_AMOUNT_MON_01'.format(cat)
        print name
        mask = (df[numerator].notnull()) & (df[denominator].notnull()) & (
         df[denominator] != 0)
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, numerator],
         df.loc[mask, denominator])

    print 'accleration and simple stats - salary'
    sal_cols = ['CSF_SAL_MON_0{}'.format(i) for i in range(1, 4)]
    for key, fn in SIMPLE.iteritems():
        name = 'CSF_salary_'+key
        print name
        df[name] = fn(df[sal_cols], axis=1)

    for col in sal_cols[1:]:
        print col
        mask = (df[col].notnull()) & (df[sal_cols[0]].notnull()) & (
         df[col] != 0)
        name = sal_cols[0]+'_RATIO'+col.strip('CSF')
        print name
        df.loc[mask, name] = map(
         lambda x, y: 1.*x/y, df.loc[mask, sal_cols[0]], df.loc[mask, col])

    df.reset_index(drop=True, inplace=True)
    return df


def sample_split(data, split=0.3, random_state=28102017):
    df = data.copy()
    features = [x for x in list(df.columns) if x.startswith('CSF')]
    x_train, x_test, y_train, y_test = train_test_split(
     df[features+['CUSTOMER_ID']], df[DV], test_size=split,
     random_state=random_state)
    # dev and val samples
    dev = pd.concat([x_train, y_train], axis=1)
    dev['sample'] = 'dev'
    val = pd.concat([x_test, y_test], axis=1)
    val['sample'] = 'val'
    cols = features + [DV, 'sample', 'CUSTOMER_ID']
    df1 = pd.concat([dev[cols], val[cols]], axis=0)
    df1.reset_index(drop=True, inplace=True)
    return df1


def preprocess(data, test, steps):
    """
    imputation, outlier treatment and scaling
    """
    df = data.copy()
    oot = test.copy()
    other_cols = ['sample', 'CUSTOMER_ID']
    features = [x for x in list(df.columns) if x.startswith('CSF')]
    classic_steps = steps
    steps = list(zip(steps, map(PREPROCESS.get, steps)))
    datapipe = Pipeline(steps=steps)
    # dev data
    dev = df[df['sample'] == 'dev']
    dev.reset_index(drop=True, inplace=True)
    # remove features that are completely missing in dev
    feats_all_missing = [
     x for x in features if dev[x].isnull().sum() == dev.shape[0]]
    print 'removed %d features as they are completely missing in dev:' % (len(
     feats_all_missing))
    features = list(set(features) - set(feats_all_missing))
    x_dev = dev[features].values
    y_dev = dev[DV].values.astype(float)
    print 'fit'
    datapipe.fit(x_dev, y_dev)
    print 'transform dataframe using pipeline'
    print 'train data:'
    train = datapipe.transform(df[features].values)
    train = pd.DataFrame(train, columns=features)
    train = pd.concat([train, df[other_cols+[DV]]], axis=1)
    print 'oot data:'
    oot1 = datapipe.transform(oot[features].values)
    oot1 = pd.DataFrame(oot1, columns=features)
    oot1 = pd.concat([oot1, oot[['CUSTOMER_ID']]], axis=1)
    # Create "classic" datapipe and store list of features
    classic_pipe = Pipeline([(name, datapipe.named_steps[name])
                             for name in classic_steps])
    classic_pipe.feature_names = features

    return train, oot1, classic_pipe


def feat_sel_cluster(data, target_col=DV, corrcoef_cutoff=0.5):
    """
    feature selection using clustering
    returns list of features
    """
    df = data.copy()
    other_cols = ['sample', 'CUSTOMER_ID', 'uid']
    feature_names = [x for x in list(df.columns) if x not in other_cols]
    dev = df[df['sample'] == 'dev'][feature_names]
    dev.reset_index(drop=True, inplace=True)
    feats = cluster_features(df=dev, target_col=target_col,
                             corrcoef_cutoff=corrcoef_cutoff)
    features = list(feats.index)
    return features


def modelfit(alg, dtrain, dtest, doot, predictors, target, performCV=True,
             cv_folds=4):
    # Fit the algorithm on the data
    print 'fit'
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Predict on testing set:
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]

    # Predict on oot set:
    doot_predprob = alg.predict_proba(doot[predictors])[:, 1]

    if performCV:
        print 'Perform cross-validation:'
        cv_score = cross_validation.cross_val_score(
         alg, dtrain[predictors], dtrain[target], cv=cv_folds,
         scoring='roc_auc')

    # Print model report:
    print "\nModel Report"
    print "Dev AUC Score : %f" % metrics.roc_auc_score(
     dtrain[target], dtrain_predprob)
    print "Val AUC Score : %f" % metrics.roc_auc_score(
     dtest[target], dtest_predprob)
    print "Dev Log Loss : %f" % metrics.log_loss(
     dtrain[target], dtrain_predprob)
    print "Val Log Loss : %f" % metrics.log_loss(
     dtest[target], dtest_predprob)

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    return alg, dtrain_predprob, dtest_predprob, doot_predprob


def get_cuts(data, bins, target):
    df = data.copy()
    table = df.groupby(bins)[target].min()
    cuts = table.unique().tolist()
    cuts.sort()
    return cuts


def apply_cuts(data, cuts, target):
    df = data.copy()
    temp = df[target]
    binned = pd.Series([-2] * len(temp), index=temp.index)
    binned[temp.isnull()] = -1
    binned[temp < np.min(cuts)] = 0

    for ibin, (low, high) in enumerate(zip(cuts[:-1], cuts[1:])):
        mask = (temp >= low) & (temp < high)
        binned[mask] = ibin + 1
    binned[temp >= np.max(cuts)] = len(cuts)
    return binned


def lift_table(data, bins, target_col):
    df = data.copy()
    table = df.groupby(bins).agg({target_col: [len, np.sum]})
    table.reset_index(inplace=True)
    table.columns = ['prob_deciles', '#total', '#responders']
    table['#non-responders'] = table['#total'] - table['#responders']
    table['perc_responders'] = table['#responders'].apply(lambda x: x/float(
     table['#responders'].sum()))
    table['perc_population'] = table['#total'].apply(lambda x: x/float(
     table['#total'].sum()))
    table['lift@decile'] = map(lambda x, y: x/float(y),
                               table['perc_responders'],
                               table['perc_population'])
    table.sort('prob_deciles', ascending=False, inplace=True)
    return table


def optimal_cutoff(dev_data, val_data, score_col):
    dev = dev_data.copy()
    val = val_data.copy()
    # get cuts from dev sample
    dev[score_col+'_binned'] = pd.qcut(dev[score_col], 10, labels=False)
    cuts = get_cuts(dev, bins=score_col+'_binned', target=score_col)
    val[score_col+'_binned'] = apply_cuts(val, cuts, target=score_col)
    table_val = lift_table(val, score_col+'_binned', DV)
    # best cutoff
    for i in range(10, 40, 1):
        val[score_col+'_pred_class_{}'.format(i)] = map(
         lambda x: 1 if x > (i/1000.) else 0, val[score_col])
    for item in [score_col+'_pred_class_{}'.format(i) for i in range(10, 40)]:
        table = pd.crosstab(val[item], val[DV])
        print table, '\n'
        tpr = table.loc[1, 1]/float(table.loc[0, 1] + table.loc[1, 1])
        fpr = table.loc[1, 0]/float(table.loc[0, 0] + table.loc[1, 0])
        precision = table.loc[1, 1]/float(table.loc[1, 0] + table.loc[1, 1])
        lift = lift_table(val, score_col+'_binned', item)
        print 'lift table: '
        print lift
        print 'tpr ', tpr
        print 'fpr ', fpr
        print 'precision ', precision


if __name__ == '__main__':
    print 'data preparation:'
    train = read_datafile(train_path)
    test = read_datafile(test_path)
    # DV
    train[DV] = train[DV].map(dv_map)
    print train.shape, train['CUSTOMER_ID'].nunique()
    print test.shape, test['CUSTOMER_ID'].nunique()
    # event rate
    print train[DV].value_counts()
    print sum(train[DV])/float(train.shape[0])
    total_bad = train[DV].sum()
    total_good = train.shape[0] - total_bad
    # check whether every customer in both train and test are salaried
    print train['OCCUP_ALL_NEW'].value_counts()
    print test['OCCUP_ALL_NEW'].value_counts()
    # convert date_cols to datetime format
    for col in date_cols:
        print col
        train[col] = train[col].apply(convert_to_date)
        test[col] = test[col].apply(convert_to_date)
    # add maturity columns
    train1 = add_maturity_cols(train)
    test1 = add_maturity_cols(test)
    print train1.shape, test1.shape
    print train1.columns
    # drop date columns
    train1.drop(date_cols, axis=1, inplace=True)
    test1.drop(date_cols, axis=1, inplace=True)
    # impute nulls in columns with 1 unique cnt
    unique_cols = [x for x in list(train1.columns) if (
     train1[x].nunique() == 1) & (x not in cols_remove)]
    print len(unique_cols)
    # print unique_cols, train1[unique_cols].dtypes
    train1 = impute_nulls(train1, feat_cols=unique_cols)
    test1 = impute_nulls(test1, feat_cols=unique_cols)

    # get summary_stats
    types = train1.dtypes
    obj_dtype = types[types == object].index.tolist()
    num_dtype = types[types != object].index.tolist()
    for col in cols_remove+[DV]:
        if col in obj_dtype:
            obj_dtype.remove(col)
        if col in num_dtype:
            num_dtype.remove(col)

    summary_cont = get_summary_stat_numeric(train1, feat_cols=num_dtype)
    print summary_cont['auc'].describe()
    summary_cat = IV_categorical(train1, feat_cols=obj_dtype)
    print summary_cat.shape
    print summary_cat.head()
    print summary_cat['IV'].describe()
    print summary_cat
    # save results
    summary_cont.to_csv(interim_path+'summary_stats_cont_vars.csv',
                        index=False)
    summary_cat.to_csv(interim_path+'summary_stats_cat_vars.csv', index=False)

    # impute categorical features with WOE values
    cat_feats = summary_cat['Feature'].unique().tolist()
    train2 = train1.copy()
    test2 = test1.copy()
    print len(cat_feats)
    for col in cat_feats:
        print col
        table = get_woe(train2[col], train2[DV])
        map_dict = dict(zip(table[col], table['WOE']))
        print map_dict
        train2[col] = train2[col].map(map_dict)
        test2[col] = test2[col].map(map_dict)

    # check dtypes
    types_train = train2.dtypes
    types_test = test2.dtypes
    print types_train[types_train == object], types_test[types_test == object]

    # impute missing values in cat_feats
    cat_cols_missing = [x for x in cat_feats if train2[x].isnull().sum() > 0]
    for col in cat_cols_missing:
        print col
        mask = train2[col].isnull()
        if train2.loc[mask, DV].sum() > 0:
            num_bad = train2.loc[mask, DV].sum()
            num_good = mask.sum() - num_bad
            perc_bad = num_bad/float(total_bad)
            perc_good = num_good/float(total_good)
            train2.loc[mask, col] = log(perc_good / float(perc_bad))
            print 'imputation in test set:'
            mask = test2[col].isnull()
            test2.loc[mask, col] = log(perc_good / float(perc_bad))

    # prefix to features
    feat_cols = get_feature_cols(train2)
    feat_cols_new = ['CSF_'+x for x in feat_cols]
    train2.rename(columns=dict(zip(feat_cols, feat_cols_new)), inplace=True)
    test2.rename(columns=dict(zip(feat_cols, feat_cols_new)), inplace=True)

    # sampling
    df = sample_split(train2)
    df.to_csv(interim_path+prepared_data_file, index=False)
    test2.to_csv(interim_path+'prepared_test_data.csv', index=False)

    # featue engineering
    df1 = create_addl_features(df)
    test3 = create_addl_features(test2)

    df1.to_csv(interim_path+'feature_engineering/train_data_engineered.csv',
               index=False)
    test3.to_csv(interim_path+'feature_engineering/test_data_engineered.csv',
                 index=False)

    # preprocessing
    df_pre, test_pre, pipeline = preprocess(df1, test3, steps=steps)
    # save
    df_pre.to_pickle(interim_path+'feature_engineering/'+'train_data_engineered_preprocessed.pkl')
    test_pre.to_pickle(interim_path+'feature_engineering/'+'test_data_engineered_preprocessed.pkl')
    pickle.dump(pipeline, open(interim_path+'feature_engineering/'+'pipeline.pkl', 'w'))

    # add unique index column for gridsearch
    df_pre['uid'] = range(len(df_pre))

    # feature selection
    # clustering
    other_cols = ['sample', 'CUSTOMER_ID', 'uid']
    feats = []
    for c in [0.5, 0.6, 0.7, 0.8]:
        feats.append(feat_sel_cluster(df_pre, corrcoef_cutoff=c))

    df_pre_c50 = df_pre[feats[0]+other_cols+[DV]]
    test_pre_c50 = test_pre[feats[0]+['CUSTOMER_ID']]
    df_pre_c60 = df_pre[feats[1]+other_cols+[DV]]
    test_pre_c60 = test_pre[feats[1]+['CUSTOMER_ID']]
    df_pre_c70 = df_pre[feats[2]+other_cols+[DV]]
    test_pre_c70 = test_pre[feats[2]+['CUSTOMER_ID']]
    df_pre_c80 = df_pre[feats[2]+other_cols+[DV]]
    test_pre_c80 = test_pre[feats[2]+['CUSTOMER_ID']]
    # save
    df_pre_c50.to_pickle(interim_path+'feature_engineering/'+featsel_out_file.format('c50'))
    test_pre_c50.to_pickle(interim_path+'feature_engineering/'+featsel_out_test_file.format('c50'))
    df_pre_c60.to_pickle(interim_path+'feature_engineering/'+featsel_out_file.format('c60'))
    test_pre_c60.to_pickle(interim_path+'feature_engineering/'+featsel_out_test_file.format('c60'))
    df_pre_c70.to_pickle(interim_path+'feature_engineering/'+featsel_out_file.format('c70'))
    test_pre_c70.to_pickle(interim_path+'feature_engineering/'+featsel_out_test_file.format('c70'))
    df_pre_c80.to_pickle(interim_path+'feature_engineering/'+featsel_out_file.format('c80'))
    test_pre_c80.to_pickle(interim_path+'feature_engineering/'+featsel_out_test_file.format('c80'))

    # gridsearch
    mask = df_pre_c80['sample'] == 'dev'
    dev = df_pre_c80.loc[mask, :]
    val = df_pre_c80.loc[~mask, :]
    dev.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    oot = test_pre_c80.copy()
    predictors = [x for x in list(df_pre_c80.columns) if x.startswith('CSF')]
    gbm_tuned = GradientBoostingClassifier(
     learning_rate=0.03, n_estimators=3000,
     max_depth=4, min_samples_split=10, min_samples_leaf=30, subsample=1.0,
     random_state=54545454, max_features='sqrt')
    modelfit(gbm_tuned, df_pre_c80, val, oot, predictors, target=DV,
             performCV=False)
    # save gbm_tuned object
    pickle.dump(gbm_tuned, open(
     interim_path+'feature_engineering/gbc_c80_alldata.pkl', 'w'))
    # score samples
    cols_needed = ['CUSTOMER_ID', DV]
    oot['pred_prob'] = gbm_tuned.predict_proba(oot[predictors])[:, 1]
    dev['pred_prob'] = gbm_tuned.predict_proba(dev[predictors])[:, 1]
    val['pred_prob'] = gbm_tuned.predict_proba(val[predictors])[:, 1]
    df_pre_c80['pred_prob'] = gbm_tuned.predict_proba(
     df_pre_c80[predictors])[:, 1]

    # find the best cutoff so as to optimize lift in the first tier
    # 0.024 leads to highest % responders and TPR
    oot[DV] = oot['pred_prob'].apply(lambda x: 1 if x > 0.03 else 0)
    submission = oot[cols_needed]

    # save results
    oot.to_csv(interim_path+'feature_engineering/predictions_gbc_c80_test.csv',
               index=False)
    df_pre_c80.to_csv(interim_path+'feature_engineering/predictions_gbc_c80_alldata.csv',
                      index=False)
    submission.to_csv(interim_path+'feature_engineering/submission.csv', index=False)

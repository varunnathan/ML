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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics

# GLOBALS
local_data_root = '/Users/vnathan/Documents/analyticsvidhya/churn/'
train_path = local_data_root + 'train.csv'
test_path = local_data_root + 'test.csv'
interim_path = local_data_root + 'intermediate/'
prepared_train_file = 'train_data_engineered.csv'
prepared_test_file = 'test_data_engineered.csv'
preprocess_train_file = 'train_data_engineered_preprocessed.pkl'
preprocess_test_file = 'test_data_engineered_preprocessed.pkl'
featsel_train_file = 'train_data_engineered_preprocessed_{}.pkl'
featsel_test_file = 'test_data_engineered_preprocessed_{}.pkl'
DV = 'Responders'
sample_col = 'sample'
cust_id = 'UCIC_ID'
cols_remove = [cust_id]
date_cols = ['AGRI_DATE', 'AL_CNC_DATE', 'AL_DATE', 'BL_DATE', 'CE_DATE',
             'CV_DATE', 'EDU_DATE', 'GL_DATE', 'LAP_DATE',
             'LAS_DATE', 'OTHER_LOANS_DATE', 'PL_DATE', 'TL_DATE', 'TWL_DATE']
date_cols = ['CSF_'+x for x in date_cols]
feat_groups = ('CSF')
PREPROCESS = {
    'exoutscaler': LegacyOutlierScaler(),
    'exaucimputer': MaxAUCImputerCV(),
    'stdscaler': StandardScaler()
}
steps = ['exaucimputer', 'exoutscaler', 'stdscaler']
SIMPLE = {'min': np.nanmin, 'max': np.nanmax, 'sum': np.nansum,
          'average': np.nanmean, 'perc50': np.nanmedian, 'std': np.nanstd}
# debit amt
d_prev_amt = ['CSF_D_prev{}'.format(i) for i in range(1, 7)]
# credit amt
c_prev_amt = ['CSF_C_prev{}'.format(i) for i in range(1, 7)]
# debit cnt
d_prev_cnt = ['CSF_count_C_prev{}'.format(i) for i in range(1, 7)]
# credit cnt
c_prev_cnt = ['CSF_count_D_prev{}'.format(i) for i in range(1, 7)]
# ATM total amt
atm_total = ['CSF_ATM_amt_prev{}'.format(i) for i in range(1, 7)]
# ATM credit amt
atm_credit_amt = ['CSF_ATM_C_prev{}'.format(i) for i in range(1, 7)]
# ATM debit amt
atm_debit_amt = ['CSF_ATM_D_prev{}'.format(i) for i in range(1, 7)]
# ATM credit count
atm_credit_cnt = ['CSF_COUNT_ATM_C_prev{}'.format(i) for i in range(1, 7)]
# ATM debit count
atm_debit_cnt = ['CSF_COUNT_ATM_D_prev{}'.format(i) for i in range(1, 7)]
# ATM cash withdrawal amt
atm_cw_amt = ['CSF_ATM_CW_Amt_prev{}'.format(i) for i in range(1, 7)]
# ATM cash withdrawal count
atm_cw_cnt = ['CSF_ATM_CW_Cnt_prev{}'.format(i) for i in range(1, 7)]
# average monthly balance
avg_mon_bal = ['CSF_BAL_prev{}'.format(i) for i in range(1, 7)]
# branch credit amt
brn_credit_amt = ['CSF_BRANCH_C_prev{}'.format(i) for i in range(1, 7)]
# branch debit amt
brn_debit_amt = ['CSF_BRANCH_D_prev{}'.format(i) for i in range(1, 7)]
# branch credit count
brn_credit_cnt = ['CSF_COUNT_BRANCH_C_prev{}'.format(i)
                  for i in range(1, 7)]
# branch debit count
brn_debit_cnt = ['CSF_COUNT_BRANCH_D_prev{}'.format(i)
                 for i in range(1, 7)]
# branch cash deposit amt
brn_cd_amt = ['CSF_BRN_CASH_Dep_Amt_prev{}'.format(i) for i in range(1, 7)]
# branch cash deposit count
brn_cd_cnt = ['CSF_BRN_CASH_Dep_Cnt_prev{}'.format(i) for i in range(1, 7)]
# branch cash withdrawal amt
brn_cw_amt = ['CSF_BRN_CW_Amt_prev{}'.format(i) for i in range(1, 7)]
# branch cash withdrawal count
brn_cw_cnt = ['CSF_BRN_CW_Cnt_prev{}'.format(i) for i in range(1, 7)]
# CASH_WD_AMT_Last6 and CASH_WD_CNT_Last6
# customer net revenue
cnr_amt = ['CSF_CNR_prev{}'.format(i) for i in range(1, 7)]
# Complaint_Logged_PrevQ1 and Complaint_Resolved_PrevQ1
# Query_Logged_PrevQ1, Query_Resolved_PrevQ1
# Dmat_Investing_PrevQ1, Dmat_Investing_PrevQ2
# FD_AMOUNT_BOOK_PrevQ1, FD_AMOUNT_BOOK_PrevQ2
# count_No_of_MF_PrevQ1, count_No_of_MF_PrevQ2
# Total_Invest_in_MF_PrevQ1, Total_Invest_in_MF_PrevQ2
# I_AQB_PrevQ1, I_AQB_PrevQ2, I_CNR_PrevQ1, I_CNR_PrevQ2
# I_CR_AQB_PrevQ1, I_CR_AQB_PrevQ2, I_NRV_PrevQ1, I_NRV_PrevQ2
# NO_OF_CHEQUE_BOUNCE_V1, NO_OF_COMPLAINTS
# NO_OF_FD_BOOK_PrevQ1, NO_OF_FD_BOOK_PrevQ2
# NO_OF_RD_BOOK_PrevQ1, NO_OF_RD_BOOK_PrevQ2
# RD_AMOUNT_BOOK_PrevQ1, RD_AMOUNT_BOOK_PrevQ2
# Req_Logged_PrevQ1, Req_Resolved_PrevQ1
# phone banking credit count
pb_credit_cnt = ['CSF_COUNT_IB_C_prev{}'.format(i) for i in range(1, 7)]
# phone banking debit count
pb_debit_cnt = ['CSF_COUNT_IB_D_prev{}'.format(i) for i in range(1, 7)]
# phone banking credit amt
pb_credit_amt = ['CSF_IB_C_prev{}'.format(i) for i in range(1, 7)]
# phone banking debit amt
pb_debit_amt = ['CSF_IB_D_prev{}'.format(i) for i in range(1, 7)]
# mobile banking credit count
mb_credit_cnt = ['CSF_COUNT_MB_C_prev{}'.format(i) for i in range(1, 7)]
# mobile banking debit count
mb_debit_cnt = ['CSF_COUNT_MB_D_prev{}'.format(i) for i in range(1, 7)]
# mobile banking credit amt
mb_credit_amt = ['CSF_MB_C_prev{}'.format(i) for i in range(1, 7)]
# mobile banking debit amt
mb_debit_amt = ['CSF_MB_D_prev{}'.format(i) for i in range(1, 7)]
# POS credit count
pos_credit_cnt = ['CSF_COUNT_POS_C_prev{}'.format(i) for i in range(1, 7)]
# POS debit count
pos_debit_cnt = ['CSF_COUNT_POS_D_prev{}'.format(i) for i in range(1, 7)]
# POS credit amt
pos_credit_amt = ['CSF_POS_C_prev{}'.format(i) for i in range(1, 7)]
# POS debit amt
pos_debit_amt = ['CSF_POS_D_prev{}'.format(i) for i in range(1, 7)]
# change in credit amb
cng_credit_amb = ['CSF_CR_AMB_Drop_Build_{}'.format(i) for i in range(1, 6)]
# CASA avg balance
casa_avg_bal = ['CSF_CR_AMB_Prev{}'.format(i) for i in range(1, 7)]
# cust initiated credit amt
custinit_credit_amt = ['CSF_custinit_CR_amt_prev{}'.format(i)
                       for i in range(1, 7)]
# cust initiated credit count
custinit_credit_cnt = ['CSF_custinit_CR_cnt_prev{}'.format(i)
                       for i in range(1, 7)]
# cust initiated debit amt
custinit_debit_amt = ['CSF_custinit_DR_amt_prev{}'.format(i)
                      for i in range(1, 7)]
# cust initiated credit count
custinit_debit_cnt = ['CSF_custinit_DR_cnt_prev{}'.format(i)
                      for i in range(1, 7)]
# end of month balance
eop_bal = ['CSF_EOP_prev{}'.format(i) for i in range(1, 7)]
# recency
recency = ['CSF_Recency_of_Activity', 'CSF_Recency_of_ATM_TXN',
           'CSF_Recency_of_BRANCH_TXN', 'CSF_Recency_of_CR_TXN',
           'CSF_Recency_of_DR_TXN', 'CSF_Recency_of_IB_TXN',
           'CSF_Recency_of_MB_TXN', 'CSF_Recency_of_POS_TXN']


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


def impute_nulls_cat_vars(data, feat_cols):
    df = data.copy()
    for col in feat_cols:
        print col
        mask = df[col].isnull()
        df.loc[mask, col] = 'missing'
        print pd.crosstab(df[col], train1[DV])

    return df


def make_flags_from_date_cols(data, feat_cols):
    df = data.copy()
    for col in feat_cols:
        print col
        mask = df[col].notnull()
        df.loc[mask, col+'_flag'] = 1
        df.loc[~mask, col+'_flag'] = 0

    return df


def get_ratio_cols(a, b):
    mask = (a.notnull()) & (b.notnull()) & (b != 0)
    series = pd.Series([np.NaN]*len(a))
    series[mask] = map(lambda x, y: 1.*x/y, a[mask], b[mask])
    return series


def discretize(data, col_name, top=20):
    df = data.copy()
    table = pd.DataFrame(df[col_name].value_counts())
    table.reset_index(inplace=True)
    print table
    df[col_name+'_trans'] = 'others'
    values = table['index'][:top].tolist()
    mask = (df[col_name].isin(values)) & (df[col_name].notnull())
    df.loc[mask, col_name+'_trans'] = df.loc[mask, col_name]
    mask = df[col_name].isnull()
    df.loc[mask, col_name+'_trans'] = 'missing'
    # transform
    tmp = df[[col_name, col_name+'_trans']]
    tmp.drop_duplicates(inplace=True)
    trans_dict = dict(zip(tmp[col_name], tmp[col_name+'_trans']))
    return df, trans_dict


def get_trans_dict_cat_vars(data, col_name):
    df = data.copy()
    table = get_woe(df[col_name], df[DV])
    table['bad_rate'] = map(
     lambda x, y: 1.*x/(x+y), table['#bad'], table['#good'])
    final_worth_dict = dict(zip(table[col_name], table['bad_rate']))
    return final_worth_dict


def feat_engg(data, final_worth_dict):
    df = data.copy()
    # simple stats on numeric features
    cols = [
     d_prev_amt, d_prev_cnt, c_prev_amt, c_prev_cnt, atm_total, atm_credit_amt,
     atm_credit_cnt, atm_debit_amt, atm_debit_cnt, atm_cw_amt, atm_cw_cnt,
     avg_mon_bal, brn_cd_amt, brn_cd_cnt, brn_credit_amt, brn_credit_cnt,
     brn_debit_amt, brn_debit_cnt, brn_cw_amt, brn_cw_cnt, cnr_amt,
     pb_credit_amt, pb_credit_cnt, pb_debit_amt, pb_debit_cnt, mb_credit_amt,
     mb_credit_cnt, mb_debit_amt, mb_debit_cnt, pos_credit_amt, pos_credit_cnt,
     pos_debit_amt, pos_debit_cnt, cng_credit_amb, casa_avg_bal,
     custinit_credit_amt, custinit_credit_cnt, custinit_debit_amt,
     custinit_debit_cnt, eop_bal, recency]
    cols_name = [
     'd_prev_amt', 'd_prev_cnt', 'c_prev_amt', 'c_prev_cnt', 'atm_total',
     'atm_credit_amt',
     'atm_credit_cnt', 'atm_debit_amt', 'atm_debit_cnt', 'atm_cw_amt',
     'atm_cw_cnt',
     'avg_mon_bal', 'brn_cd_amt', 'brn_cd_cnt', 'brn_credit_amt',
     'brn_credit_cnt',
     'brn_debit_amt', 'brn_debit_cnt', 'brn_cw_amt', 'brn_cw_cnt', 'cnr_amt',
     'pb_credit_amt', 'pb_credit_cnt', 'pb_debit_amt', 'pb_debit_cnt',
     'mb_credit_amt',
     'mb_credit_cnt', 'mb_debit_amt', 'mb_debit_cnt', 'pos_credit_amt',
     'pos_credit_cnt',
     'pos_debit_amt', 'pos_debit_cnt', 'cng_credit_amb', 'casa_avg_bal',
     'custinit_credit_amt', 'custinit_credit_cnt', 'custinit_debit_amt',
     'custinit_debit_cnt', 'eop_bal', 'recency'
    ]
    cols_dict = dict(zip(cols_name, cols))
    for name, value in cols_dict.iteritems():
        for key, fn in SIMPLE.iteritems():
            col = 'CSF_'+name+'_'+key
            print col
            df[col] = fn(df[value], axis=1)

    # CV
    std_cols = ['CSF_'+x+'_std' for x in cols_name]
    avg_cols = ['CSF_'+x+'_average' for x in cols_name]
    for i, col in enumerate(std_cols):
        a = df[col]
        b = df[avg_cols[i]]
        name = col + '_RATIO_' + avg_cols[i]
        df[name] = get_ratio_cols(a, b)

    # Complaint_Resolved_PrevQ1
    correct_map = {'1.0': 1, 1: 1, '2.0': 2, 2: 2, '3.0': 3, 3: 3, '4.0': 4,
                   4: 4, '5.0': 5, 5: 5, '6.0': 6, 6: 6, '7.0': 7, 7: 7,
                   '8.0': 8, 8: 8, '9.0': 9, 9: 9, '0.0': 0, 0: 0,
                   'missing': 'missing', '>': '9+'}
    df['CSF_Complaint_Resolved_PrevQ1'] = df[
     'CSF_Complaint_Resolved_PrevQ1'].map(correct_map)
    # Query_Resolved_PrevQ1
    df['CSF_Query_Resolved_PrevQ1'] = df[
     'CSF_Query_Resolved_PrevQ1'].map(correct_map)
    # query-complaint ratio
    df['CSF_Query_Complaint_Ratio'] = get_ratio_cols(
     df['CSF_Complaint_Logged_PrevQ1'], df['CSF_Query_Logged_PrevQ1'])
    # Req_Resolved_PrevQ1
    df['CSF_Req_Resolved_PrevQ1_map'] = df['CSF_Req_Resolved_PrevQ1'].map(
     correct_map)
    # Dmat ratio
    df['CSF_Dmat_Investing_Ratio'] = get_ratio_cols(
     df['CSF_Dmat_Investing_PrevQ1'], df['CSF_Dmat_Investing_PrevQ2'])
    # FD
    df['CSF_FD_AMOUNT_BOOK_Ratio'] = get_ratio_cols(
     df['CSF_FD_AMOUNT_BOOK_PrevQ1'], df['CSF_FD_AMOUNT_BOOK_PrevQ2'])
    df['CSF_FD_AMOUNT_PrevQ1_Avg'] = get_ratio_cols(
     df['CSF_FD_AMOUNT_BOOK_PrevQ1'], df['CSF_NO_OF_FD_BOOK_PrevQ1'])
    df['CSF_FD_AMOUNT_PrevQ2_Avg'] = get_ratio_cols(
     df['CSF_FD_AMOUNT_BOOK_PrevQ2'], df['CSF_NO_OF_FD_BOOK_PrevQ2'])
    df['CSF_FD_AMOUNT_BOOK_sum'] = df['CSF_FD_AMOUNT_BOOK_PrevQ1'] + df[
     'CSF_FD_AMOUNT_BOOK_PrevQ2']
    # RD
    df['CSF_RD_AMOUNT_BOOK_Ratio'] = get_ratio_cols(
     df['CSF_RD_AMOUNT_BOOK_PrevQ1'], df['CSF_RD_AMOUNT_BOOK_PrevQ2'])
    df['CSF_RD_AMOUNT_PrevQ1_Avg'] = get_ratio_cols(
     df['CSF_RD_AMOUNT_BOOK_PrevQ1'], df['CSF_NO_OF_RD_BOOK_PrevQ1'])
    df['CSF_RD_AMOUNT_PrevQ2_Avg'] = get_ratio_cols(
     df['CSF_RD_AMOUNT_BOOK_PrevQ2'], df['CSF_NO_OF_RD_BOOK_PrevQ2'])
    df['CSF_RD_AMOUNT_BOOK_sum'] = df['CSF_RD_AMOUNT_BOOK_PrevQ1'] + df[
     'CSF_RD_AMOUNT_BOOK_PrevQ2']
    # MF
    df['CSF_MF_Invest_Ratio'] = get_ratio_cols(
     df['CSF_Total_Invest_in_MF_PrevQ1'], df['CSF_Total_Invest_in_MF_PrevQ2'])
    df['CSF_MF_Invest_PrevQ1_Avg'] = get_ratio_cols(
     df['CSF_Total_Invest_in_MF_PrevQ1'], df['CSF_count_No_of_MF_PrevQ1'])
    df['CSF_MF_Invest_PrevQ2_Avg'] = get_ratio_cols(
     df['CSF_Total_Invest_in_MF_PrevQ2'], df['CSF_count_No_of_MF_PrevQ2'])
    # I_AQB
    df['CSF_I_AQB_Ratio'] = get_ratio_cols(
     df['CSF_I_AQB_PrevQ1'], df['CSF_I_AQB_PrevQ2'])
    df['CSF_I_AQB_sum'] = df['CSF_I_AQB_PrevQ1'] + df['CSF_I_AQB_PrevQ2']
    # I_CNR
    df['CSF_I_CNR_Ratio'] = get_ratio_cols(
     df['CSF_I_CNR_PrevQ1'], df['CSF_I_CNR_PrevQ1'])
    df['CSF_I_CNR_sum'] = df['CSF_I_CNR_PrevQ1'] + df['CSF_I_CNR_PrevQ1']
    # I_CR_AQB
    df['CSF_I_CR_AQB_Ratio'] = get_ratio_cols(
     df['CSF_I_CR_AQB_PrevQ1'], df['CSF_I_CR_AQB_PrevQ2'])
    df['CSF_I_CR_AQB_sum'] = df['CSF_I_CR_AQB_PrevQ1'] + df[
     'CSF_I_CR_AQB_PrevQ2']
    # I_NRV
    df['CSF_I_NRV_Ratio'] = get_ratio_cols(
     df['CSF_I_NRV_PrevQ1'], df['CSF_I_NRV_PrevQ2'])
    df['CSF_I_NRV_sum'] = df['CSF_I_NRV_PrevQ1'] + df[
     'CSF_I_NRV_PrevQ2']

    # use cat variables

    for name, value in cols_dict.iteritems():
        for key, fn in SIMPLE.iteritems():
            col = 'CSF_'+name+'_'+key
            col_name = col + '_product_final_worth'
            print col_name
            df[col_name] = map(
             lambda x, y: final_worth_dict[x]*y, df['CSF_FINAL_WORTH_prev1'],
             df[col])

    return df


def feat_engg_subsample(data, engagement_tag_dict):
    df = data.copy()
    df['CSF_d_prev_amt_sum_ratio_avg_mon_bal_sum'] = get_ratio_cols(
     df['CSF_d_prev_amt_sum'], df['CSF_avg_mon_bal_sum'])
    df['CSF_atm_credit_amt_sum_ratio_c_prev_amt_sum'] = get_ratio_cols(
     df['CSF_atm_credit_amt_sum'], df['CSF_c_prev_amt_sum'])
    df['CSF_atm_credit_amt_std_ratio_c_prev_amt_std'] = get_ratio_cols(
     df['CSF_atm_credit_amt_std'], df['CSF_c_prev_amt_std'])
    df['CSF_brn_credit_amt_sum_ratio_c_prev_amt_sum'] = get_ratio_cols(
     df['CSF_brn_credit_amt_sum'], df['CSF_c_prev_amt_sum'])
    df['CSF_brn_debit_amt_sum_ratio_d_prev_amt_sum'] = get_ratio_cols(
     df['CSF_brn_debit_amt_sum'], df['CSF_d_prev_amt_sum'])
    df['CSF_pos_credit_amt_sum_ratio_c_prev_amt_sum'] = get_ratio_cols(
     df['CSF_pos_credit_amt_sum'], df['CSF_c_prev_amt_sum'])
    df['CSF_pb_credit_amt_sum_ratio_c_prev_amt_sum'] = get_ratio_cols(
     df['CSF_pb_credit_amt_sum'], df['CSF_c_prev_amt_sum'])
    df['CSF_mb_credit_amt_sum_ratio_c_prev_amt_sum'] = get_ratio_cols(
     df['CSF_mb_credit_amt_sum'], df['CSF_c_prev_amt_sum'])
    df['CSF_custinit_credit_amt_sum_ratio_c_prev_amt_sum'] = get_ratio_cols(
     df['CSF_custinit_credit_amt_sum'], df['CSF_c_prev_amt_sum'])

    # range
    cols_name = [
     'd_prev_amt', 'd_prev_cnt', 'c_prev_amt', 'c_prev_cnt', 'atm_total',
     'atm_credit_amt',
     'atm_credit_cnt', 'atm_debit_amt', 'atm_debit_cnt', 'atm_cw_amt',
     'atm_cw_cnt',
     'avg_mon_bal', 'brn_cd_amt', 'brn_cd_cnt', 'brn_credit_amt',
     'brn_credit_cnt',
     'brn_debit_amt', 'brn_debit_cnt', 'brn_cw_amt', 'brn_cw_cnt', 'cnr_amt',
     'pb_credit_amt', 'pb_credit_cnt', 'pb_debit_amt', 'pb_debit_cnt',
     'mb_credit_amt',
     'mb_credit_cnt', 'mb_debit_amt', 'mb_debit_cnt', 'pos_credit_amt',
     'pos_credit_cnt',
     'pos_debit_amt', 'pos_debit_cnt', 'cng_credit_amb', 'casa_avg_bal',
     'custinit_credit_amt', 'custinit_credit_cnt', 'custinit_debit_amt',
     'custinit_debit_cnt', 'eop_bal', 'recency'
    ]
    cols_max = ['CSF_'+x+'_max' for x in cols_name]
    cols_min = ['CSF_'+x+'_min' for x in cols_name]
    for i, col in enumerate(cols_max):
        name = col+'_diff_'+cols_min[i]
        df[name] = df[col] - df[cols_min[i]]

    cols = ['CSF_eop_bal_std_RATIO_CSF_eop_bal_average',
            'CSF_casa_avg_bal_std_RATIO_CSF_casa_avg_bal_average',
            'CSF_avg_mon_bal_std_RATIO_CSF_avg_mon_bal_average',
            'CSF_avg_mon_bal_std', 'CSF_casa_avg_bal_std',
            'CSF_eop_bal_std', 'CSF_CR_AMB_Prev3', 'CSF_BAL_prev3',
            'CSF_casa_avg_bal_max', 'CSF_avg_mon_bal_max',
            'CSF_Percent_Change_in_FT_outside', 'CSF_cng_credit_amb_std']
    for col in cols:
        df[col+'_product_ENGAGEMENT_TAG'] = map(
          lambda x, y: 1.*engagement_tag_dict[x]*y,
          df['CSF_ENGAGEMENT_TAG_prev1'], df[col])

    return df


def sample_split(data, split=0.3, random_state=06112017):
    df = data.copy()
    features = [x for x in list(df.columns) if x.startswith('CSF')]
    x_train, x_test, y_train, y_test = train_test_split(
     df[features+[cust_id]], df[DV], test_size=split,
     random_state=random_state)
    # dev and val samples
    dev = pd.concat([x_train, y_train], axis=1)
    dev['sample'] = 'dev'
    val = pd.concat([x_test, y_test], axis=1)
    val['sample'] = 'val'
    cols = features + [DV, 'sample', cust_id]
    df1 = pd.concat([dev[cols], val[cols]], axis=0)
    df1.reset_index(drop=True, inplace=True)
    return df1


def preprocess(data, test, steps):
    """
    imputation, outlier treatment and scaling
    """
    df = data.copy()
    oot = test.copy()
    other_cols = ['sample', cust_id]
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
    oot1 = pd.concat([oot1, oot[[cust_id]]], axis=1)
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
    other_cols = ['sample', cust_id, 'uid']
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
    for i in range(260, 350, 1):
        val[score_col+'_pred_class_{}'.format(i)] = map(
         lambda x: 1 if x > (i/1000.) else 0, val[score_col])
    for item in [score_col+'_pred_class_{}'.format(i) for i in range(260, 350)]:
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
    print 'churn prediction...'
    print 'data preparation:'
    train = read_datafile(train_path)
    test = read_datafile(test_path)

    train1 = train.copy()
    test1 = test.copy()
    types = train1.dtypes
    obj_dtype = types[types == object].index.tolist()
    num_dtype = types[types != object].index.tolist()
    for col in cols_remove+[DV]:
        if col in obj_dtype:
            obj_dtype.remove(col)
        if col in num_dtype:
            num_dtype.remove(col)
    print len(types)
    print len(obj_dtype)
    print len(num_dtype)

    # impute nulls in categorical vars with 'missing'
    train1 = impute_nulls_cat_vars(train1, feat_cols=obj_dtype)
    test1 = impute_nulls_cat_vars(test1, feat_cols=obj_dtype)

    # add prefix
    feat_cols = [x for x in list(train1.columns) if x not in [cust_id, DV]]
    print len(feat_cols)
    feat_cols_new = ['CSF_'+x for x in feat_cols]
    train1.rename(columns=dict(zip(feat_cols, feat_cols_new)), inplace=True)
    test1.rename(columns=dict(zip(feat_cols, feat_cols_new)), inplace=True)

    # flags for date columns
    train1 = make_flags_from_date_cols(train1, feat_cols=date_cols)
    test1 = make_flags_from_date_cols(test1, feat_cols=date_cols)

    # remove date_cols
    train1.drop(date_cols, axis=1, inplace=True)
    test1.drop(date_cols, axis=1, inplace=True)

    # feature engineering
    final_worth_dict = get_trans_dict_cat_vars(
     train1, col_name='CSF_FINAL_WORTH_prev1')
    train2 = feat_engg(train1, final_worth_dict)
    test2 = feat_engg(test1, final_worth_dict)

    # brn_code, city, CSF_dependents, NO_OF_Accs, vintage, zip
    train2, brn_code_trans = discretize(train2, col_name='CSF_brn_code')
    train2, city_trans = discretize(train2, col_name='CSF_city')
    train2, dep_trans = discretize(train2, col_name='CSF_dependents', top=5)
    train2, accs_trans = discretize(train2, col_name='CSF_NO_OF_Accs', top=3)
    train2, vintage_trans = discretize(train2, col_name='CSF_vintage')
    train2, zip_trans = discretize(train2, col_name='CSF_zip')

    test2['CSF_brn_code_trans'] = test2['CSF_brn_code'].map(brn_code_trans)
    test2['CSF_city_trans'] = test2['CSF_brn_code'].map(city_trans)
    test2['CSF_dependents_trans'] = test2['CSF_brn_code'].map(dep_trans)
    test2['CSF_NO_OF_Accs_trans'] = test2['CSF_brn_code'].map(accs_trans)
    test2['CSF_vintage_trans'] = test2['CSF_brn_code'].map(vintage_trans)
    test2['CSF_zip_trans'] = test2['CSF_brn_code'].map(zip_trans)

    # remove brn_code, city, CSF_dependents, NO_OF_Accs, vintage, zip
    cols = ['CSF_brn_code', 'CSF_city', 'CSF_dependents', 'CSF_NO_OF_Accs',
            'CSF_vintage', 'CSF_zip']
    train2.drop(cols, axis=1, inplace=True)
    test2.drop(cols, axis=1, inplace=True)

    # impute categorical columns with WOE values
    types = train2.dtypes
    cat_feats = types[types == object].index.tolist()
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

    # sampling
    df = sample_split(train2)
    df.to_csv(interim_path+prepared_train_file, index=False)
    test2.to_csv(interim_path+prepared_test_file, index=False)

    # preprocessing
    df_pre, test_pre, pipeline = preprocess(df, test2, steps=steps)

    # save
    df_pre.to_pickle(interim_path+preprocess_train_file)
    test_pre.to_pickle(interim_path+preprocess_test_file)
    pickle.dump(pipeline, open(interim_path+'pipeline.pkl', 'w'))

    # add unique index column for gridsearch
    df_pre['uid'] = range(len(df_pre))

    # feature selection
    # clustering
    other_cols = ['sample', cust_id, 'uid']
    feats = []
    for c in [0.6, 0.7]:
        feats.append(feat_sel_cluster(df_pre, corrcoef_cutoff=c))

    df_pre_c60 = df_pre[feats[0]+other_cols+[DV]]
    test_pre_c60 = test_pre[feats[0]+[cust_id]]
    df_pre_c70 = df_pre[feats[1]+other_cols+[DV]]
    test_pre_c70 = test_pre[feats[1]+[cust_id]]

    # save
    df_pre_c60.to_csv(interim_path+featsel_train_file.format('C60'))
    test_pre_c60.to_csv(interim_path+featsel_test_file.format('C60'))
    df_pre_c70.to_csv(interim_path+featsel_train_file.format('C70'))
    test_pre_c70.to_csv(interim_path+featsel_test_file.format('C70'))

    # gridsearch
    mask = df_pre_c70['sample'] == 'dev'
    dev = df_pre_c70.loc[mask, :]
    val = df_pre_c70.loc[~mask, :]
    dev.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    oot = test_pre_c70.copy()
    predictors = [x for x in list(df_pre_c70.columns) if x.startswith('CSF')]
    gbm_tuned = GradientBoostingClassifier(
     learning_rate=0.03, n_estimators=3000,
     max_depth=4, min_samples_split=30, min_samples_leaf=10, subsample=1.0,
     random_state=54545454, max_features='sqrt')
    model, dev_prob, val_prob, oot_prob = modelfit(
     gbm_tuned, dev, val, oot, predictors, target=DV, performCV=False)

    # save model object
    pickle.dump(model, open(interim_path+'gbc_c70_dev_best.pkl', 'w'))

    # score samples
    cols_needed = [cust_id, DV]
    oot['pred_prob'] = oot_prob
    dev['pred_prob'] = dev_prob
    val['pred_prob'] = val_prob
    df_pre_c70['pred_prob'] = model.predict_proba(
     df_pre_c70[predictors])[:, 1]

    # find the best cutoff so as to optimize lift in the first two tiers
    # 0.267 leads to highest % responders and TPR
    oot[DV] = oot['pred_prob'].apply(lambda x: 1 if x > 0.267 else 0)
    submission = oot[cols_needed]

    # save
    submission.to_csv(interim_path+'submission_1_Nov6.csv', index=False)

    # build a model on top 30% population of responders
    dev['pred_prob_binned'] = pd.qcut(dev['pred_prob'], 10, labels=False)
    cuts = get_cuts(dev, bins='pred_prob_binned', target='pred_prob')
    df_pre_c70['pred_prob_binned'] = apply_cuts(
     df_pre_c70, cuts, target='pred_prob')
    oot['pred_prob_binned'] = apply_cuts(oot, cuts, target='pred_prob')
    df_pre_c70_top30 = df_pre_c70[
     df_pre_c70['pred_prob_binned'].isin([10, 9, 8])]
    df_pre_c70_top30.reset_index(drop=True, inplace=True)

    # filter raw data (df)
    accs = df_pre_c70_top30[cust_id].unique().tolist()
    df_top30 = df[df[cust_id].isin(accs)]
    df_top30.reset_index(drop=True, inplace=True)
    test3 = pd.merge(test2, oot[[cust_id, 'pred_prob', 'pred_prob_binned']],
                     on=cust_id)

    # engineer more features
    table = get_woe(df_top30['CSF_ENGAGEMENT_TAG_prev1'], df_top30[DV])
    table['bad_rate'] = map(lambda x, y: 1.*x/(x+y), table['#bad'],
                            table['#good'])
    print table
    engagement_tag_dict = dict(zip(table['CSF_ENGAGEMENT_TAG_prev1'],
                                   table['bad_rate']))
    df_top30_1 = feat_engg_subsample(df_top30, engagement_tag_dict)
    test3 = feat_engg_subsample(test3, engagement_tag_dict)

    # preprocessing
    df_top30_pre, test3_pre, pipeline = preprocess(
     df_top30, test3, steps=steps)
    # add unique index column for gridsearch
    df_top30_pre['uid'] = range(len(df_top30_pre))
    # feature selection
    # clustering
    other_cols = ['sample', cust_id, 'uid']
    feats = []
    for c in [0.7]:
        feats.append(feat_sel_cluster(df_top30_pre, corrcoef_cutoff=c))

    df_top30_pre_c70 = df_top30_pre[feats[0]+other_cols+[DV]]
    test3_pre_c70 = test3_pre[feats[0]+[cust_id]]

    # gridsearch
    mask = df_top30_pre_c70['sample'] == 'dev'
    dev1 = df_top30_pre_c70.loc[mask, :]
    val1 = df_top30_pre_c70.loc[~mask, :]
    dev1.reset_index(drop=True, inplace=True)
    val1.reset_index(drop=True, inplace=True)
    oot1 = test3_pre_c70.copy()
    predictors = [x for x in list(df_top30_pre_c70.columns) if x.startswith('CSF')]
    gbm_tuned = GradientBoostingClassifier(
     learning_rate=0.03, n_estimators=2000, min_weight_fraction_leaf=0.0,
     max_depth=4, min_samples_split=10, min_samples_leaf=10, subsample=1.0,
     random_state=54545454, max_features='sqrt')
    model, dev_prob, val_prob, oot_prob = modelfit(
     gbm_tuned, dev1, val1, oot1, predictors, target=DV, performCV=False)

    # save model object
    pickle.dump(model, open(interim_path+'gbc_c70_dev_best_top30.pkl', 'w'))

    # score samples
    cols_needed = [cust_id, DV]
    oot1['pred_prob_top30'] = oot_prob
    dev1['pred_prob_top30'] = dev_prob
    val1['pred_prob_top30'] = val_prob
    df_top30_pre_c70['pred_prob_top30'] = model.predict_proba(
     df_top30_pre_c70[predictors])[:, 1]

    # combine
    train_final = pd.merge(
     df_pre_c70, df_top30_pre_c70[[cust_id, 'pred_prob_top30']],
     on=cust_id, how='left')
    mask = train_final['pred_prob_top30'].notnull()
    train_final.loc[mask, 'pred_prob_final'] = train_final.loc[mask, 'pred_prob_top30']
    train_final.loc[~mask, 'pred_prob_final'] = train_final.loc[~mask, 'pred_prob']
    test_final = pd.merge(
     oot, oot1[[cust_id, 'pred_prob_top30']], on=cust_id, how='left')
    mask = test_final['pred_prob_binned'].isin([10, 9, 8])
    test_final.loc[mask, 'pred_prob_final'] = test_final.loc[mask, 'pred_prob_top30']
    test_final.loc[~mask, 'pred_prob_final'] = test_final.loc[~mask, 'pred_prob']

    # find the best cutoff so as to optimize lift in the first two tiers based on val sample
    mask = train_final['sample'] == 'val'
    dev = train_final.loc[~mask, :]
    val = train_final.loc[mask, :]
    dev.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    optimal_cutoff(dev, val, score_col='pred_prob_final')

    # 0.31 leads to highest % responders and TPR
    test_final[DV] = test_final['pred_prob_final'].apply(
     lambda x: 1 if x > 0.325 else 0)
    submission = test_final[cols_needed]

    # save
    submission.to_csv(interim_path+'submission_2_Nov6.csv', index=False)

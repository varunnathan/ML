# Imports

import pandas as pd
import numpy as np

import os, sys, time, json, copy, re, joblib, argparse

from collections import deque, defaultdict

# To compute similarities between vectors
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# To use recommender systems
from surprise import (Reader, Dataset, SVD, SVDpp,
                      KNNBaseline, BaselineOnly)
from surprise.model_selection import cross_validate

# To create deep learning models
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model

# To create sparse matrices
from scipy.sparse import coo_matrix, csr_matrix
from scipy import sparse

# To light fm
from lightfm import LightFM
from lightfm.evaluation import precision_at_k

# To stack sparse matrices
from scipy.sparse import vstack

from read_utils import (load_single_user_file, load_qualifying_data)
from constants import *
from baseline_feat_calc_utils import *
from neighbourhood_feat_calc_utils import *
from surpriseMF_feat_calc_utils import SurpriseMF
from lightfm_feat_calc_utils import (
    get_user_list, get_item_list, id_mappings, get_interaction_matrix,
    lightfm_train_utils, lightfm_eval_utils, recommendation_sampling)


# GLOBALS
parser = argparse.ArgumentParser()
parser.add_argument('task', choices=["prepare-data", "feature-calc"],
                    help="task to perform")
parser.add_argument('--file-num', type=int, default=1, help='file number of user data')
parser.add_argument('--segment', type=str, default='None', help='User-Movie segment')
parser.add_argument('--feature-type', choices=['baseline', 'neighbourhood',
                                             'surpriseMF', 'lightFM'],
                    default='baseline', help='feature type')
parser.add_argument('--data-type', choices=['user_data', 'qualifying_data',
                                          'segment_data'], default='user_data',
                    help='data type for preparation')
parser.add_argument('--feature-task', choices=['training', 'validation', 'both'],
                    default='both', help='feature calculation task')

args = parser.parse_args()
TASK = args.task
FILE_NUM = args.file_num
SEGMENT = args.segment
FEATURE_TYPE = args.feature_type
DATA_TYPE = args.data_type
FEATURE_TASK = args.feature_task


def prepare_user_data(file_nums=FILE_NUMS):

    start = time.time()
    for file_num in file_nums:

        print('file num: %d' % (file_num))

        out_fn = os.path.join(OUT_DIR, 'user_data_{}.h5'.format(file_num))

        if not os.path.isfile(out_fn):
            out = load_single_user_file(file_num)
            print('save\n')
            out.to_hdf(out_fn, key='stage', mode='w')

            print('time taken for file_num %d: %0.2f' % (file_num, time.time()-start))

        else:
            print('file %s already exists' % (out_fn))
            continue

    print('total time taken: %0.2f' % (time.time() - start))


def main_neighbourhood_feats_calc(file_num, df_train, num_segments,
                                  num_similar):

    start = time.time()
    print('define segments \n')
    segments = define_segments_for_modelling(df_train, num_segments)
    print('time taken: %0.2f' % (time.time()))

    print('number of segments: %d' % (len(segments)))
    print('\n')

    test_rmses = []
    user_movies_ratings_global_dct = {}
    movie_segment_dct = defaultdict(list)
    for user_bin, movie_bin in segments:
        print('User Bin: %s and Movie Bin: %s' % (str(user_bin),
                                                  str(movie_bin)))
        (sim_movie_dct, user_movies_ratings_dct,
         df_train_segment_train, df_train_segment_test,
         rmse) = segment_modelling(df_train, user_bin, movie_bin,
                                   0.1, num_similar)

        print('update movie_segment_dct \n')
        movie_ids = list(sim_movie_dct.keys())
        for movie_id in movie_ids:
            if movie_id not in movie_segment_dct:
                movie_segment_dct[movie_id].append((user_bin, movie_bin))
            elif (user_bin, movie_bin) not in movie_segment_dct[movie_id]:
                movie_segment_dct[movie_id].append((user_bin, movie_bin))

        print('update user_movies_ratings_global_dct')
        for k in user_movies_ratings_dct:
            if k in user_movies_ratings_global_dct:
                user_movies_ratings_global_dct[k] += user_movies_ratings_dct[k]
            else:
                user_movies_ratings_global_dct[k] = user_movies_ratings_dct[k]

        test_rmses.append(rmse)

        print('save df_train_segment_train \n')
        df_train_segment_train.to_csv(
            SEGMENT_TRAIN_FN.format(user_bin, movie_bin, file_num),
            index=False)
        print('save df_train_segment_test \n')
        df_train_segment_test.to_csv(
            SEGMENT_TEST_FN.format(user_bin, movie_bin, file_num),
            index=False)
        print('save sim_movie_dct \n')
        json.dump(sim_movie_dct, open(
            SIM_MOVIE_DCT_FN.format(user_bin, movie_bin, file_num), 'w'))

        print('time taken: %0.2f' % (time.time()))
        print('\n')


    test_rmse_df = pd.DataFrame({'segment': segments,
                                 'test_rmse': test_rmses})
    print('save test_rmse_df \n')
    test_rmse_df.to_csv(TEST_RMSE_FN.format(file_num), index=False)

    print('save user_movies_ratings_global_dct \n')
    json.dump(user_movies_ratings_global_dct,
              open(USER_MOVIE_RATINGS_DCT_FN.format(file_num), 'w'))

    print('save movie_segment_dct \n')
    json.dump(movie_segment_dct,
              open(MOVIE_SEGMENT_DCT_FN.format(file_num), 'w'))

    print('overall time taken: %0.2f' % (time.time()))
    print('done')


def main_baseline_feats_calc(df_train):

    print('initialize feature dicts\n')

    mean_ratings_movie_dct = init_feat_dict(MEAN_RATINGS_MOVIE_DCT_FN)
    mean_ratings_user_dct = init_feat_dict(MEAN_RATINGS_USER_DCT_FN)
    weighted_mean_ratings_movie_dct = init_feat_dict(
        WEIGHTED_MEAN_RATINGS_MOVIE_DCT_FN)
    user_earliest_date_dct = init_feat_dict(EARLIEST_RATING_DATE_USER_FN)
    movie_earliest_date_dct = init_feat_dict(EARLIEST_RATING_DATE_MOVIE_FN)
    user_latest_date_dct = init_feat_dict(LATEST_RATING_DATE_USER_FN)
    movie_latest_date_dct = init_feat_dict(LATEST_RATING_DATE_MOVIE_FN)
    movie_num_ratings = init_feat_dict(NUM_RATINGS_MOVIE_DCT_FN)
    user_num_ratings = init_feat_dict(NUM_RATINGS_USER_DCT_FN)

    print('convert Date to datetime format\n')
    df_train['Date'] = pd.to_datetime(df_train['Date'], format='%Y-%m-%d')

    print('mean ratings for movie\n')
    d = baseline_feat_calc_helper(df_train, 'Movie', 'mean')
    mean_ratings_movie_dct.update(d)
    del d

    print('weighted mean ratings for movie\n')
    d = get_weighted_mean_ratings(df_train, 250, 'Movie', 'Rating')
    weighted_mean_ratings_movie_dct.update(d)
    del d

    print('movie earliest date dct\n')
    d = baseline_feat_calc_helper(df_train, 'Movie', 'earliest')
    movie_earliest_date_dct.update(d)
    del d

    print('movie latest date dct\n')
    d = baseline_feat_calc_helper(df_train, 'Movie', 'latest')
    movie_latest_date_dct.update(d)
    del d

    print('user earliest date dct\n')
    d = baseline_feat_calc_helper(df_train, 'User', 'earliest')
    user_earliest_date_dct = _update_user_dct(d, user_earliest_date_dct,
                                              'earliest')
    del d

    print('user latest date dct\n')
    d = baseline_feat_calc_helper(df_train, 'User', 'latest')
    user_latest_date_dct = _update_user_dct(d, user_latest_date_dct,
                                            'latest')
    del d

    print('movie num ratings\n')
    d = baseline_feat_calc_helper(df_train, 'Movie', 'count')
    movie_num_ratings.update(d)
    del d

    print('user num ratings\n')
    user_num_ratings_local = baseline_feat_calc_helper(
        df_train, 'User', 'count')
    user_num_ratings = _update_user_dct(user_num_ratings_local,
                                        user_num_ratings, 'count')

    print('mean ratings for user\n')
    d = baseline_feat_calc_helper(df_train, 'User', 'mean')
    mean_ratings_user_dct = _update_user_dct(
        d, mean_ratings_user_dct, user_num_ratings_local,
        user_num_ratings, 'mean')
    del d, user_num_ratings_local, df_train

    print('save outputs to disk\n')

    print('mean_ratings_movie_dct\n')
    json.dump(mean_ratings_movie_dct,
              open(MEAN_RATINGS_MOVIE_DCT_FN, 'w'))

    print('mean_ratings_user_dct\n')
    json.dump(mean_ratings_user_dct,
              open(MEAN_RATINGS_USER_DCT_FN, 'w'))

    print('weighted_mean_ratings_movie_dct\n')
    json.dump(weighted_mean_ratings_movie_dct,
              open(WEIGHTED_MEAN_RATINGS_MOVIE_DCT_FN, 'w'))

    print('user_earliest_date_dct\n')
    json.dump(user_earliest_date_dct,
              open(EARLIEST_RATING_DATE_USER_FN, 'w'))

    print('movie_earliest_date_dct\n')
    json.dump(movie_earliest_date_dct,
              open(EARLIEST_RATING_DATE_MOVIE_FN, 'w'))

    print('user_latest_date_dct\n')
    json.dump(user_latest_date_dct,
              open(LATEST_RATING_DATE_USER_FN, 'w'))

    print('movie_latest_date_dct\n')
    json.dump(movie_latest_date_dct,
              open(LATEST_RATING_DATE_MOVIE_FN, 'w'))

    print('movie_num_ratings\n')
    json.dump(movie_num_ratings, open(NUM_RATINGS_MOVIE_DCT_FN, 'w'))

    print('user_num_ratings\n')
    json.dump(user_num_ratings, open(NUM_RATINGS_USER_DCT_FN, 'w'))


def combine_sim_movie_dcts(file_num, movie_segment_dct, sim_dct,
                           top_k=150):

    sim_dct_global = {}
    count = 1
    for movie_id in movie_segment_dct:
        if count % 50 == 0:
            print('num completed: ', count)
            print('\n')
        print('Movie: ', movie_id)
        segments = ['U{}_M{}'.format(x[0],x[1])
                    for x in movie_segment_dct[movie_id]]
        print('num segments: ', len(segments))
        movie_tups = []
        for segment in segments:
            movie_tups += sim_dct[segment][movie_id]

        tmp = pd.DataFrame(movie_tups, columns=['movie_id', 'sim'])
        tmp1 = tmp.groupby('movie_id')['sim'].max().rename(
            'max_sim').reset_index()
        tmp1.sort_values('max_sim', ascending=False, inplace=True)
        tmp1 = tmp1[:top_k]
        sim_dct_global[movie_id] = list(zip(tmp1['movie_id'],
                                            tmp1['max_sim']))
        count += 1
        print('\n')

    print('save sim_dct_global')
    json.dump(sim_dct_global,
              open(SIM_MOVIE_DCT_GLOBAL_FN.format(file_num), 'w'))

    return sim_dct_global


def main_baseline_feats_calc_val(
    df, mean_ratings_movie_dct, mean_ratings_user_dct,
    weighted_mean_ratings_movie_dct, user_earliest_date_dct,
    movie_earliest_date_dct, user_latest_date_dct, movie_latest_date_dct,
    movie_num_ratings, user_num_ratings, movie_col='Movie',
    user_col='User', date_col='Date'):

    df_val = df.copy()

    print('convert Date to datetime format\n')
    df_val[date_col] = pd.to_datetime(df_val[date_col], format='%Y-%m-%d')

    print('convert to datetime format in date_dcts')
    user_earliest_date_dct = {k: pd.to_datetime(v, format='%Y-%m-%d')
                              for k, v in user_earliest_date_dct.items()}
    movie_earliest_date_dct = {k: pd.to_datetime(v, format='%Y-%m-%d')
                               for k, v in movie_earliest_date_dct.items()}
    user_latest_date_dct = {k: pd.to_datetime(v, format='%Y-%m-%d')
                            for k, v in user_latest_date_dct.items()}
    movie_latest_date_dct = {k: pd.to_datetime(v, format='%Y-%m-%d')
                             for k, v in movie_latest_date_dct.items()}

    print('mean_ratings_movie')
    df_val['mean_ratings_movie'] = df_val[movie_col].apply(
        lambda x: mean_ratings_movie_dct.get(str(x), None))

    print('mean_ratings_user')
    df_val['mean_ratings_user'] = df_val[user_col].apply(
        lambda x: mean_ratings_user_dct.get(str(x), None))

    print('weighted_mean_ratings_movie')
    df_val['weighted_mean_ratings_movie'] = df_val[movie_col].apply(
        lambda x: weighted_mean_ratings_movie_dct.get(str(x), None))

    print('days_since_first_user_rating')
    df_val['days_since_first_user_rating'] = list(
        map(lambda user, date: (date -
                                user_earliest_date_dct[str(user)]).days
            if str(user) in user_earliest_date_dct else None,
            df_val[user_col], df_val[date_col]))

    print('sqrt_days_since_first_user_rating')
    df_val['sqrt_days_since_first_user_rating'] = df_val[
        'days_since_first_user_rating'].apply(
        lambda x: np.sqrt(x) if x is not None else None)
    mask = df_val['days_since_first_user_rating'] < 0
    df_val.loc[mask, 'sqrt_days_since_first_user_rating'] = 0

    print('rating_age_days_user')
    df_val['rating_age_days_user'] = list(
        map(lambda user: (user_latest_date_dct[str(user)] -
                          user_earliest_date_dct[str(user)]).days
            if (str(user) in user_earliest_date_dct) and
            (str(user) in user_latest_date_dct) else None,
            df_val[user_col]))

    print('rating_age_weeks_user')
    df_val['rating_age_weeks_user'] = df_val['rating_age_days_user'].apply(
        lambda x: x/7. if x is not None else None)

    print('rating_age_months_user')
    df_val['rating_age_months_user'] = df_val['rating_age_days_user'].apply(
        lambda x: x/30. if x is not None else None)

    print('days_since_first_movie_rating')
    df_val['days_since_first_movie_rating'] = list(
        map(lambda movie, date: (date -
                                 movie_earliest_date_dct[str(movie)]).days
            if str(movie) in movie_earliest_date_dct else None,
            df_val[movie_col], df_val[date_col]))

    print('sqrt_days_since_first_movie_rating')
    df_val['sqrt_days_since_first_movie_rating'] = df_val[
        'days_since_first_movie_rating'].apply(
        lambda x: np.sqrt(x) if x is not None else None)
    mask = df_val['days_since_first_movie_rating'] < 0
    df_val.loc[mask, 'sqrt_days_since_first_movie_rating'] = 0

    print('rating_age_days_movie')
    df_val['rating_age_days_movie'] = list(
        map(lambda movie: (movie_latest_date_dct[str(movie)] -
                           movie_earliest_date_dct[str(movie)]).days
            if (str(movie) in movie_earliest_date_dct) and
            (str(movie) in movie_latest_date_dct) else None,
            df_val[movie_col]))

    print('rating_age_weeks_movie')
    df_val['rating_age_weeks_movie'] = df_val['rating_age_days_movie'].apply(
        lambda x: x/7. if x is not None else None)

    print('rating_age_months_movie')
    df_val['rating_age_months_movie'] = df_val['rating_age_days_movie'].apply(
        lambda x: x/30. if x is not None else None)

    print('num_ratings_movie')
    df_val['num_ratings_movie'] = df_val[movie_col].apply(
        lambda x: movie_num_ratings.get(str(x), None))

    print('num_ratings_user')
    df_val['num_ratings_user'] = df_val[user_col].apply(
        lambda x: user_num_ratings.get(str(x), None))

    return df_val


def main_neighbourhood_feats_calc_val(file_num, df, movie_col='Movie',
                                      user_col='User', date_col='Date'):

    df_val = df.copy()

    if os.path.isfile(SIM_MOVIE_DCT_GLOBAL_FN.format(file_num)):
        print('sim_movie_dct_global exists. Reading from disk...')
        sim_movie_dct_global = json.load(open(
            SIM_MOVIE_DCT_GLOBAL_FN.format(file_num)))
    else:
        print('identify segments on which training was done')
        sw, ew = 'sim_movie_dct_', '_{}.json'.format(file_num)
        files = [x for x in os.listdir(NEIGHBOURHOOD_FEATS_DIR) if
                 x.startswith(sw) and x.endswith(ew)]
        segments = [x.split(sw)[-1].split(ew)[0] for x in files]

        print('number of segments: ', len(segments))

        print('read sim_movie_dcts')
        sim_dct = {}
        for i, file in enumerate(files):
            segment = segments[i]
            fn = os.path.join(NEIGHBOURHOOD_FEATS_DIR, file)
            sim_dct[segment] = json.load(open(fn))

        print('read movie_segment_dct')
        movie_segment_dct = json.load(open(
            MOVIE_SEGMENT_DCT_FN.format(file_num)))

        print('combine sim_movie_dcts')
        sim_movie_dct_global = combine_sim_movie_dcts(
            file_num, movie_segment_dct, sim_dct, top_k=150)

    print('read user_movie_ratings_dct')
    user_movies_ratings_dct = json.load(open(
        USER_MOVIE_RATINGS_DCT_FN.format(file_num)))

    print('calc weighted score based on sim_movie_dct_global')
    scores = calc_features_test_df(df_val, sim_movie_dct_global,
                                   user_movies_ratings_dct)
    feat_name = 'pred_score_item_item_neighbourhood_model'
    df_val[feat_name] = scores

    print('shape: ', df_val.shape)
    return df_val


def main_surpriseMF_feats_calc(file_num, segment, calc_feats_val=True):
    start = time.time()

    print('read segment train and test sets\n')
    user_bin, movie_bin = re.findall('[0-9]+', segment)
    train_fn = SEGMENT_TRAIN_FN.format(user_bin, movie_bin, file_num)
    test_fn = SEGMENT_TEST_FN.format(user_bin, movie_bin, file_num)
    df_train = pd.read_csv(train_fn)
    df_test = pd.read_csv(test_fn)
    print('time taken: %0.2f' % (time.time() - start))

    print('Instantiate training class\n')
    surprise_cls = SurpriseMF(algorithm='svd', choose_algorithm_flag=False)

    print('Training\n')
    surprise_cls.fit(df_train)
    print('time taken: %0.2f' % (time.time() - start))

    del df_train

    print('Prediction on df_test_segment\n')
    pred_scores = surprise_cls.predict(df_test)
    df_test['pred_rating_surpriseSVD'] = pred_scores
    del pred_scores
    print('time taken: %0.2f' % (time.time() - start))

    print('save df_test_segment\n')
    df_test.to_csv(SURPRISEMF_SEGMENT_TEST_FN.format(user_bin, movie_bin,
                                                     file_num), index=False)

    print('RMSE calculation\n')
    rmse = np.sqrt(mean_squared_error(y_true=df_test['Rating'],
                                      y_pred=df_test['pred_rating_surpriseSVD']))
    test_rmse_df = pd.DataFrame({'segment': segment, 'test_rmse': rmse},
                                index=range(1))

    if os.path.isfile(SURPRISEMF_TEST_RMSE_FN.format(file_num)):
        rmse_df = pd.read_csv(SURPRISEMF_TEST_RMSE_FN.format(file_num))
        cols = ['segment', 'test_rmse']
        rmse_df = pd.concat([rmse_df[cols], test_rmse_df[cols]], axis=0)
        rmse_df.reset_index(drop=True, inplace=True)
    else:
        rmse_df = test_rmse_df.copy()

    rmse_df.to_csv(SURPRISEMF_TEST_RMSE_FN.format(file_num), index=False)
    del df_test, rmse_df, test_rmse_df
    print('time taken: %0.2f' % (time.time() - start))

    print('save fitted class object')
    joblib.dump(surprise_cls, open(
        SURPRISEMF_CLASS_FN.format(user_bin, movie_bin, file_num), 'wb'))
    print('total time taken: %0.2f' % (time.time() - start))

    if calc_feats_val:
        print('calculate features on val sample\n')

        if os.path.isfile(FEATS_VAL_FN.format(file_num)):
            print('features data for val exist. Reading from disk\n')
            df_val = pd.read_hdf(FEATS_VAL_FN.format(file_num), key='stage')
        else:
            print('reading val interim data from disk\n')
            df_val = pd.read_hdf(VAL_FN.format(file_num), key='stage')

        print('Prediction on df_val\n')
        pred_scores = surprise_cls.predict(df_val)
        df_val['pred_rating_surpriseSVD_U{}_M{}'.format(user_bin, movie_bin)] = pred_scores
        del pred_scores

        print('save\n')
        df_val.to_hdf(FEATS_VAL_FN.format(file_num), key='stage', mode='w')
        del df_val
        print('time taken: %0.2f' % (time.time() - start))

        print('\n')

        print('calculate features on test sample\n')

        if os.path.isfile(FEATS_TEST_FN.format(file_num)):
            print('features data for test exist. Reading from disk\n')
            df_test = pd.read_hdf(FEATS_TEST_FN.format(file_num), key='stage')
        else:
            print('reading test interim data from disk\n')
            df_test = pd.read_hdf(TEST_FN.format(file_num), key='stage')

        print('Prediction on df_test\n')
        pred_scores = surprise_cls.predict(df_test)
        df_test['pred_rating_surpriseSVD_U{}_M{}'.format(user_bin, movie_bin)] = pred_scores
        del pred_scores

        print('save\n')
        df_test.to_hdf(FEATS_TEST_FN.format(file_num), key='stage', mode='w')
        print('time taken: %0.2f' % (time.time() - start))


def main_surpriseMF_feats_calc_val(file_num, segment):
    start = time.time()

    print('read fitted svd class')
    user_bin, movie_bin = re.findall('[0-9]+', segment)
    surprise_cls = joblib.load(SURPRISEMF_CLASS_FN.format(user_bin, movie_bin,
                                                          file_num))

    print('calculate features on val sample\n')

    if os.path.isfile(FEATS_VAL_FN.format(file_num)):
        print('features data for val exist. Reading from disk\n')
        df_val = pd.read_hdf(FEATS_VAL_FN.format(file_num), key='stage')
    else:
        print('reading val interim data from disk\n')
        df_val = pd.read_hdf(VAL_FN.format(file_num), key='stage')

    print('Prediction on df_val\n')
    pred_scores = surprise_cls.predict(df_val)
    df_val['pred_rating_surpriseSVD_U{}_M{}'.format(user_bin, movie_bin)] = pred_scores
    del pred_scores

    print('save\n')
    df_val.to_hdf(FEATS_VAL_FN.format(file_num), key='stage', mode='w')
    del df_val
    print('time taken: %0.2f' % (time.time() - start))

    print('\n')

    print('calculate features on test sample\n')

    if os.path.isfile(FEATS_TEST_FN.format(file_num)):
        print('features data for test exist. Reading from disk\n')
        df_test = pd.read_hdf(FEATS_TEST_FN.format(file_num), key='stage')
    else:
        print('reading test interim data from disk\n')
        df_test = pd.read_hdf(TEST_FN.format(file_num), key='stage')

    print('Prediction on df_test\n')
    pred_scores = surprise_cls.predict(df_test)
    df_test['pred_rating_surpriseSVD_U{}_M{}'.format(user_bin, movie_bin)] = pred_scores
    del pred_scores

    print('save\n')
    df_test.to_hdf(FEATS_TEST_FN.format(file_num), key='stage', mode='w')
    print('time taken: %0.2f' % (time.time() - start))


def main_lightfm_feats_calc(file_num, no_components=20, loss='warp', epochs=1,
                            eval_k=20, calc_feats_val=True):

    print('read data')
    df = pd.read_hdf(INP_FN.format(file_num), key='stage')

    print('create the user and item lists')
    users = get_user_list(df, "User")
    items = get_item_list(df, "Movie")

    print('# users: ', len(users))
    print('# movies: ', len(items))

    print('generate mapping')
    (user_to_index_mapping, index_to_user_mapping,
     item_to_index_mapping, index_to_item_mapping) = id_mappings(users, items)

    print('generate user_item_interaction_matrix for df')
    user_to_product_interaction = get_interaction_matrix(
        df, "User", "Movie", "Rating", user_to_index_mapping,
        item_to_index_mapping)

    print('save artifacts for future use')
    joblib.dump(user_to_product_interaction,
                open(LIGHTFM_INTERACTION_FN.format(file_num), 'wb'))
    json.dump(user_to_index_mapping,
              open(LIGHTFM_USER_2_INDEX_MAP_FN.format(file_num), 'w'))
    json.dump(item_to_index_mapping,
              open(LIGHTFM_ITEM_2_INDEX_MAP_FN.format(file_num), 'w'))

    del df

    print('read train data')
    df_train = pd.read_hdf(TRAIN_FN.format(file_num), key='stage')

    print('read val data')
    df_val = pd.read_hdf(VAL_FN.format(file_num), key='stage')

    print('generate user_item_interaction_matrix for train data')
    user_to_product_interaction_train = get_interaction_matrix(
        df_train, "User", "Movie", "Rating", user_to_index_mapping,
        item_to_index_mapping)

    print('generate user_item_interaction_matrix for test data')
    user_to_product_interaction_test = get_interaction_matrix(
        df_val, "User", "Movie", "Rating", user_to_index_mapping,
        item_to_index_mapping)

    print(user_to_product_interaction_train.shape)
    print(user_to_product_interaction_test.shape)

    print('Training')
    kwargs = {'no_components': no_components, 'loss': loss, 'epochs': epochs,
              'model_out_fn': LightFM_MODEL_FN.format(file_num)}
    model = lightfm_train_utils(user_to_product_interaction_train, **kwargs)

    print('Evaluation')
    lightfm_eval_utils(user_to_product_interaction_test, model,
                       **{'eval_k': eval_k})

    if calc_feats_val:
        print('Prediction')
        print('read val features data')
        df_val = pd.read_hdf(FEATS_VAL_FN.format(file_num), key='stage')

        print('read test features data')
        df_test = pd.read_hdf(FEATS_TEST_FN.format(file_num), key='stage')

        recom = recommendation_sampling(
            model=model, items=items,
            user_to_product_interaction_matrix=user_to_product_interaction,
            user2index_map=user_to_index_mapping,
            item2index_map=item_to_index_mapping)

        print('prediction on val data')
        user_lst = df_val['User'].tolist()
        movie_lst = df_val['Movie'].tolist()
        scores = recom.get_score(user_lst, movie_lst)
        df_val['pred_score_lightfm'] = scores

        print('prediction on test data')
        user_lst = df_test['User'].tolist()
        movie_lst = df_test['Movie'].tolist()
        scores = recom.get_score(user_lst, movie_lst)
        df_test['pred_score_lightfm'] = scores

        print('save')
        df_val.to_hdf(FEATS_VAL_FN.format(file_num), key='stage', mode='w')
        df_test.to_hdf(FEATS_TEST_FN.format(file_num), key='stage', mode='w')


def main_lightfm_feats_calc_val(file_num):

    print('Prediction')
    print('read val features data')
    df_val = pd.read_hdf(FEATS_VAL_FN.format(file_num), key='stage')

    print('read test features data')
    df_test = pd.read_hdf(FEATS_TEST_FN.format(file_num), key='stage')

    print('read model object')
    model = joblib.load(LightFM_MODEL_FN.format(file_num))

    print('read user_to_product_interaction matrix')
    user_to_product_interaction = joblib.load(LIGHTFM_INTERACTION_FN.format(file_num))

    print('read user_to_index_mapping')
    user_to_index_mapping = json.load(open(LIGHTFM_USER_2_INDEX_MAP_FN.format(file_num)))

    print('read item_to_index_mapping')
    item_to_index_mapping = json.load(open(LIGHTFM_ITEM_2_INDEX_MAP_FN.format(file_num)))

    print('get item list')
    items = list(item_to_index_mapping.keys())

    recom = recommendation_sampling(
        model=model, items=items,
        user_to_product_interaction_matrix=user_to_product_interaction,
        user2index_map=user_to_index_mapping,
        item2index_map=item_to_index_mapping)

    print('prediction on val data')
    user_lst = df_val['User'].tolist()
    movie_lst = df_val['Movie'].tolist()
    scores = recom.get_score(user_lst, movie_lst)
    df_val['pred_score_lightfm'] = scores

    print('prediction on test data')
    user_lst = df_test['User'].tolist()
    movie_lst = df_test['Movie'].tolist()
    scores = recom.get_score(user_lst, movie_lst)
    df_test['pred_score_lightfm'] = scores

    print('save')
    df_val.to_hdf(FEATS_VAL_FN.format(file_num), key='stage', mode='w')
    df_test.to_hdf(FEATS_TEST_FN.format(file_num), key='stage', mode='w')


def main_feats_calc(file_num, task='training', feat_type=None,
                    num_user_bins=15, num_movie_bins=10,
                    num_segments=4, num_similar=150):
    """
    feat_type: neighbourhood, baseline
    task: training, validation
    """

    if task == 'training':
        print('Task: ', task)
        start_time = time.time()

        if not os.path.isfile(TRAIN_FN.format(file_num)):
            print('init \n')
            df = _initialize(file_num, INP_FN, num_user_bins, num_movie_bins)

            print('overall sampling \n')
            df_train, df_val, df_test = sampling(df, split=0.01,
                                                 _type='overall')

            print('save df_train \n')
            df_train.to_hdf(TRAIN_FN.format(file_num), key='stage',
                            mode='w')

            print('save df_val \n')
            df_val.to_hdf(VAL_FN.format(file_num), key='stage', mode='w')

            print('save df_test \n')
            df_test.to_hdf(TEST_FN.format(file_num), key='stage', mode='w')

        else:
            print('File exists. Reading from disk...\n')
            df_train = pd.read_hdf(TRAIN_FN.format(file_num), key='stage')

        if feat_type == 'neighbourhood':
            print('feature calculation begins...\n')
            main_neighbourhood_feats_calc(file_num, df_train, num_segments,
                                          num_similar)
            print('time taken for feature calculation: %0.2f' % (
                time.time() - start_time))

        elif feat_type == 'baseline':
            print('feature calculation begins...\n')
            main_baseline_feats_calc(df_train)
            print('time taken for feature calculation: %0.2f' % (
                time.time() - start_time))

    elif task == 'validation':
        print('Task: ', task)

        if os.path.isfile(FEATS_VAL_FN.format(file_num)):
            print('features data for val exist. Reading from disk')
            df_val = pd.read_hdf(FEATS_VAL_FN.format(file_num),
                                 key='stage')
        else:
            print('reading val interim data from disk')
            df_val = pd.read_hdf(VAL_FN.format(file_num), key='stage')

        if feat_type == 'baseline':

            print('baseline feature calculation on val\n')

            print('read feature dictionaries\n')

            print('mean_ratings_movie_dct\n')
            mean_ratings_movie_dct = json.load(
                open(MEAN_RATINGS_MOVIE_DCT_FN))

            print('mean_ratings_user_dct\n')
            mean_ratings_user_dct = json.load(open(MEAN_RATINGS_USER_DCT_FN))

            print('weighted_mean_ratings_movie_dct\n')
            weighted_mean_ratings_movie_dct = json.load(
                open(WEIGHTED_MEAN_RATINGS_MOVIE_DCT_FN))

            print('user_earliest_date_dct\n')
            user_earliest_date_dct = json.load(
                open(EARLIEST_RATING_DATE_USER_FN))

            print('movie_earliest_date_dct\n')
            movie_earliest_date_dct = json.load(
                open(EARLIEST_RATING_DATE_MOVIE_FN))

            print('user_latest_date_dct\n')
            user_latest_date_dct = json.load(
                open(LATEST_RATING_DATE_USER_FN))

            print('movie_latest_date_dct\n')
            movie_latest_date_dct = json.load(
                open(LATEST_RATING_DATE_MOVIE_FN))

            print('movie_num_ratings\n')
            movie_num_ratings = json.load(open(NUM_RATINGS_MOVIE_DCT_FN))

            print('user_num_ratings\n')
            user_num_ratings = json.load(open(NUM_RATINGS_USER_DCT_FN))

            print('feature calculation begins...\n')
            df_val = main_baseline_feats_calc_val(
                df_val, mean_ratings_movie_dct, mean_ratings_user_dct,
                weighted_mean_ratings_movie_dct, user_earliest_date_dct,
                movie_earliest_date_dct, user_latest_date_dct,
                movie_latest_date_dct, movie_num_ratings, user_num_ratings,
                movie_col='Movie', user_col='User', date_col='Date')

            print('xxxxxxxxxxxxxxxx\n\n')

        elif feat_type == 'neighbourhood':

            print('neighbourhood feature calculation on val\n')
            df_val = main_neighbourhood_feats_calc_val(
                file_num, df_val, movie_col='Movie', user_col='User',
                date_col='Date')

            print('xxxxxxxxxxxxxxxx\n\n')

        print('save df_val\n')
        df_val.to_hdf(FEATS_VAL_FN.format(file_num), key='stage',
                      mode='w')
        del df_val

        print('xxxxxxxxxxxxxxxx\n\n')

        if os.path.isfile(FEATS_TEST_FN.format(file_num)):
            print('features data for test exist. Reading from disk')
            df_test = pd.read_hdf(FEATS_TEST_FN.format(file_num),
                                  key='stage')
        else:
            print('reading test interim data from disk')
            df_test = pd.read_hdf(TEST_FN.format(file_num), key='stage')

        if feat_type == 'baseline':
            print('baseline feature calculation on test\n')

            df_test = main_baseline_feats_calc_val(
                df_test, mean_ratings_movie_dct, mean_ratings_user_dct,
                weighted_mean_ratings_movie_dct, user_earliest_date_dct,
                movie_earliest_date_dct, user_latest_date_dct,
                movie_latest_date_dct, movie_num_ratings, user_num_ratings,
                movie_col='Movie', user_col='User', date_col='Date')

            print('xxxxxxxxxxxxxxxx\n\n')

        elif feat_type == 'neighbourhood':
            print('neighbourhood feature calculation on test\n')
            df_test = main_neighbourhood_feats_calc_val(
                file_num, df_test, movie_col='Movie', user_col='User',
                date_col='Date')

            print('xxxxxxxxxxxxxxxx\n\n')

        print('save df_test\n')
        df_test.to_hdf(FEATS_TEST_FN.format(file_num), key='stage',
                       mode='w')


if __name__ == '__main__':
    print('execution begins')
    if TASK == 'prepare-data':
        if DATA_TYPE == 'user_data':
            print('User data preparation')
            prepare_user_data()
            print('done')

    elif TASK == 'feature-calc':
        print('FILE NUM: ', FILE_NUM)
        print('Feature Task: ', FEATURE_TASK)
        print('Feature Type: ', FEATURE_TYPE)
        if FEATURE_TYPE not in ('surpriseMF', 'lightFM'):
            main_feats_calc(file_num=FILE_NUM, task=FEATURE_TASK,
                            feat_type=FEATURE_TYPE)
        elif FEATURE_TYPE == 'surpriseMF':
            print('Segment: ', SEGMENT)
            if FEATURE_TASK == 'training':
                main_surpriseMF_feats_calc(FILE_NUM, SEGMENT, calc_feats_val=False)
            elif FEATURE_TASK == 'validation':
                main_surpriseMF_feats_calc_val(FILE_NUM, SEGMENT)
            elif FEATURE_TASK == 'both':
                main_surpriseMF_feats_calc(FILE_NUM, SEGMENT, calc_feats_val=True)
        elif FEATURE_TYPE == 'lightFM':
            if FEATURE_TASK == 'training':
                main_lightfm_feats_calc(FILE_NUM, calc_feats_val=False)
            elif FEATURE_TASK == 'validation':
                main_lightfm_feats_calc_val(FILE_NUM)
            elif FEATURE_TASK == 'both':
                main_lightfm_feats_calc(FILE_NUM, calc_feats_val=True)

import pandas as pd
import numpy as np
import os, json, time, sys
from constants import *

SAMPLE = sys.argv[1]
FILE_NUM = sys.argv[2]


def _find_files(_dir, start_token, end_token):
    return [x for x in os.listdir(_dir) if (x.startswith(start_token))
            and (x.endswith(end_token))]


class PrepareData(object):

    def __init__(self, sample, file_num=None, interim_data_dir=OUT_DIR,
                 prep_data_dir=PREPARED_DATA_DIR,
                 user_col='User', item_col='Movie', end_token='.h5',
                 date_col='Date', dv_col='Rating', good_ratings=[5],
                 start_token='user_{}_data_', user2idx_fn=USER2IDX_FN,
                 item2idx_fn=ITEM2IDX_FN, baseline_feats=False,
                 erd_user_fn=EARLIEST_RATING_DATE_USER_FN,
                 erd_movie_fn=EARLIEST_RATING_DATE_MOVIE_FN,
                 lrd_user_fn=LATEST_RATING_DATE_USER_FN,
                 lrd_movie_fn=LATEST_RATING_DATE_MOVIE_FN,
                 mr_user_fn=MEAN_RATINGS_USER_DCT_FN,
                 mr_movie_fn=MEAN_RATINGS_MOVIE_DCT_FN,
                 wmr_movie_fn=WEIGHTED_MEAN_RATINGS_MOVIE_DCT_FN,
                 nr_user_fn=NUM_RATINGS_USER_DCT_FN,
                 nr_movie_fn=NUM_RATINGS_MOVIE_DCT_FN):

        if file_num is None:
            files = _find_files(interim_data_dir, start_token.format(sample),
                                end_token)
            self.files = [os.path.join(interim_data_dir, x) for x in files]
            self.out_files = [os.path.join(prep_data_dir, x) for x in files]
        else:
            file_name = start_token.format(sample)+str(file_num)+end_token
            self.files = [os.path.join(interim_data_dir, file_name)]
            self.out_files = [os.path.join(prep_data_dir, file_name)]
        print(self.files)
        print(self.out_files)
        self.date_col = date_col
        self.dv_col = dv_col
        self.good_ratings = good_ratings
        self.dv_col_class = dv_col + '_class'
        self.user_col = user_col
        self.item_col = item_col
        self.user2idx = json.load(open(user2idx_fn))
        self.item2idx = json.load(open(item2idx_fn))
        self.baseline_feats = baseline_feats

        if self.baseline_feats:

            self.erd_user = self.convert_to_datetime(json.load(open(
                erd_user_fn)))
            d = self.calc_median(self.erd_user, self.user2idx,
                                 'earliest')
            if d is not None:
                self.erd_user.update(d)

            self.erd_movie = self.convert_to_datetime(json.load(open(
                erd_movie_fn)))
            d = self.calc_median(self.erd_movie, self.item2idx,
                                 'earliest')
            if d is not None:
                self.erd_movie.update(d)

            self.lrd_user = self.convert_to_datetime(json.load(open(
                lrd_user_fn)))
            d = self.calc_median(self.lrd_user, self.user2idx, 'latest')
            if d is not None:
                self.lrd_user.update(d)

            self.lrd_movie = self.convert_to_datetime(json.load(open(
                lrd_movie_fn)))
            d = self.calc_median(self.lrd_movie, self.item2idx, 'latest')
            if d is not None:
                self.lrd_movie.update(d)

            self.mr_user = json.load(open(mr_user_fn))
            d = self.calc_median(self.mr_user, self.user2idx, None)
            if d is not None:
                self.mr_user.update(d)

            self.mr_movie = json.load(open(mr_movie_fn))
            d = self.calc_median(self.mr_movie, self.item2idx, None)
            if d is not None:
                self.mr_movie.update(d)

            self.wmr_movie = json.load(open(wmr_movie_fn))
            d = self.calc_median(self.wmr_movie, self.item2idx, None)
            if d is not None:
                self.wmr_movie.update(d)

            self.nr_user = json.load(open(nr_user_fn))
            d = self.calc_median(self.nr_user, self.user2idx, None)
            if d is not None:
                self.nr_user.update(d)

            self.nr_movie = json.load(open(nr_movie_fn))
            d = self.calc_median(self.nr_movie, self.item2idx, None)
            if d is not None:
                self.nr_movie.update(d)


    def read_file(self, fn):

        df = pd.read_hdf(fn, key='stage')
        return df

    def convert_to_datetime(self, feat_dct):
        return {k: pd.to_datetime(v, format='%Y-%m-%d')
                for k, v in feat_dct.items()}

    def calc_median(self, feat_dct, ent2idx, date):

        if len(ent2idx) > len(feat_dct):
            print('there are missing users or movies')
            missing_users = set(ent2idx.keys())
            missing_users = missing_users - set(feat_dct.keys())
            if date is None:
                median = np.nanmedian(list(feat_dct.values()))
            elif date == 'earliest':
                median = np.nanmin(list(feat_dct.values()))
            elif date == 'latest':
                median = np.nanmax(list(feat_dct.values()))
            d = dict(zip(missing_users, [median]*len(missing_users)))
            return d
        else:
            return None

    def calc_date_features(self, data, erd_feat_dct, lrd_feat_dct,
                           feat_type):

        if feat_type == 'user':
            ent_col = self.user_col
        elif feat_type == 'item':
            ent_col = self.item_col

        parent_feature = 'days_since_first_{}_rating'.format(feat_type)
        print(parent_feature)
        data[parent_feature] = list(
            map(lambda user, date: (date -
                                    erd_feat_dct[str(user)]).days
                if str(user) in erd_feat_dct else None,
                data[ent_col], data[self.date_col]))

        feature = 'sqrt_days_since_first_{}_rating'.format(feat_type)
        print(feature)
        data[feature] = data[parent_feature].apply(
            lambda x: np.sqrt(x) if (x is not None) and (x>=0) else None)
        mask = data[parent_feature] < 0
        data.loc[mask, feature] = -1
        data.loc[mask, parent_feature] = -1

        feature = 'rating_age_days_{}'.format(feat_type)
        print(feature)
        data[feature] = list(
            map(lambda user: (lrd_feat_dct[str(user)] -
                              erd_feat_dct[str(user)]).days
                if (str(user) in erd_feat_dct) and
                (str(user) in lrd_feat_dct) else None, data[ent_col]))

        feature = 'rating_age_weeks_{}'.format(feat_type)
        parent_feature = 'rating_age_days_{}'.format(feat_type)
        print(feature)
        data[feature] = data[parent_feature].apply(
            lambda x: x/7. if x is not None else None)

        feature = 'rating_age_months_{}'.format(feat_type)
        parent_feature = 'rating_age_days_{}'.format(feat_type)
        print(feature)
        data[feature] = data[parent_feature].apply(
            lambda x: x/30. if x is not None else None)

        return data

    def preprocess(self, data):

        print('convert %s to datetime format' % (self.date_col))
        data[self.date_col] = pd.to_datetime(data[self.date_col],
                                             format='%Y-%m-%d')

        print('encoding for DV - only for classification experiments')
        mask = data[self.dv_col].isin(self.good_ratings)
        data[self.dv_col_class] = 0
        data.loc[mask, self.dv_col_class] = 1

        if self.baseline_feats:
            print('adding user and item baseline features\n')

            print('Define ordered tuple of feature columns\n')
            user_feats = [
                ('mean_ratings_user', self.mr_user),
                ('num_ratings_user', self.nr_user)]

            movie_feats = [
                ('mean_ratings_movie', self.mr_movie),
                ('weighted_mean_ratings_movie', self.wmr_movie),
                ('num_ratings_movie', self.nr_movie)]

            print('User Features\n')

            print('date features\n')
            data = self.calc_date_features(data, self.erd_user,
                                           self.lrd_user, 'user')
            print('other features\n')
            for feat_name, feat_dct in user_feats:
                print('Feature: ', feat_name)
                data[feat_name] = data[self.user_col].apply(
                    lambda x: feat_dct.get(str(x), None))

            print('Item Features\n')

            print('date features\n')
            data = self.calc_date_features(data, self.erd_movie,
                                           self.lrd_movie, 'item')
            print('other features\n')
            for feat_name, feat_dct in movie_feats:
                print('Feature: ', feat_name)
                data[feat_name] = data[self.item_col].apply(
                    lambda x: feat_dct.get(str(x), None))

        print('encoding for categorical variables')
        data[self.user_col] = data[self.user_col].apply(
            lambda x: self.user2idx[str(x)])
        data[self.item_col] = data[self.item_col].apply(
            lambda x: self.item2idx[str(x)])

        print('drop unwanted columns')
        data.drop(['num_rating_user', 'num_rating_user_bins',
                   'num_rating_movie', 'num_rating_movie_bins'],
                  axis=1, inplace=True)

        return data

    def prepare_data(self):

        for i, fn in enumerate(self.files):
            if os.path.isfile(self.out_files[i]):
                print('file %s already exists in the disk.' % (
                    self.out_files[i]))
                continue
            print('num completed files: %d' % (i))

            start = time.time()

            print('read data')
            data = self.read_file(fn)
            print('time taken: %0.2f' % (time.time() - start))
            print('\n\n\n')

            print('preprocess data')
            data = self.preprocess(data)
            print('time taken: %0.2f' % (time.time() - start))
            print('\n\n\n')

            print('save')
            out_fn = self.out_files[i]
            print('out file: %s' % (out_fn))
            data.to_hdf(out_fn, key='stage', mode='w')
            del data
            print('time taken: %0.2f' % (time.time() - start))
            print('\n\n\n')


if __name__ == '__main__':
    print('data preparation for NN modelling...')
    start = time.time()

    print('Instantiate class\n')
    prep_data = PrepareData(sample=SAMPLE, file_num=FILE_NUM, baseline_feats=True)
    print('time taken: %0.2f' % (time.time() - start))
    print('\n\n\n')

    print('data preparation\n')
    prep_data.prepare_data()
    print('time taken: %0.2f' % (time.time() - start))

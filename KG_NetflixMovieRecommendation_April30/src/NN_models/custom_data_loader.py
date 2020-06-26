import sys, json
import pandas as pd
from torch.utils.data import IterableDataset
from itertools import chain
sys.path.append("../")
from constants import *


class InteractionsStream(IterableDataset):

    def __init__(self, prep_data_dir=PREPARED_DATA_DIR, file_num=None,
                 sample='train', user_col='User', item_col='Movie',
                 end_token='.h5', start_token='user_{}_data_',
                 baseline_feats=False, model_type='regression',
                 chunksize=10, normalize=False, title_features=False,
                 numeric_params_fn=NUMERIC_FEATS_PARAMS_DCT_FN,
                 title_feats_fn=MOVIE_TITLES_TFIDF_COMPS_FN):

        if file_num is None:
            self.files = [os.path.join(prep_data_dir, x) for x in
                          _find_files(prep_data_dir,
                                      start_token.format(sample),
                                      end_token)]
        else:
            self.files = [
                os.path.join(prep_data_dir,
                             start_token.format(sample)+str(file_num)+
                             end_token)]
        print(self.files)
        self.user_col = user_col
        self.item_col = item_col
        self.baseline_feats = baseline_feats
        self.sample = sample
        self.chunksize = chunksize
        if model_type == 'regression':
            self.dv_col = 'Rating'
        elif model_type == 'classification':
            self.dv_col = 'Rating_class'
        self.cat_cols = [self.user_col, self.item_col]
        self.normalize = normalize
        self.title_features = title_features

        if self.normalize:
            self.numeric_params_dct = json.load(open(numeric_params_fn))

        if self.title_features:
            self.title_feats_dct = json.load(open(title_feats_fn))

        if baseline_feats:
            self.numeric_cols = [
                'days_since_first_user_rating',
                'sqrt_days_since_first_user_rating',
                'rating_age_days_user', 'rating_age_weeks_user',
                'rating_age_months_user', 'mean_ratings_user',
                'num_ratings_user', 'days_since_first_item_rating',
                'sqrt_days_since_first_item_rating',
                'rating_age_days_item', 'rating_age_weeks_item',
                'rating_age_months_item', 'mean_ratings_movie',
                'weighted_mean_ratings_movie', 'num_ratings_movie']
        else:
            self.numeric_cols = []

    def read_file(self, fn):

        if self.sample == 'train':
            df = pd.read_hdf(fn, key='stage', iterator=True,
                             chunksize=self.chunksize)
        else:
            df = pd.read_hdf(fn, key='stage')

        return df

    def transform_numeric_cols(self, numeric_params_dct, numeric_cols,
                                x):
        x_new = []
        count = 0
        for item in x:
            if isinstance(item, list):
                x_new_item = []
                for i, value in enumerate(item):
                    d = numeric_params_dct[numeric_cols[i]]
                    x_new_item.append((value - d['mean'])/d['std'])
                x_new.append(x_new_item)
            else:
                d = numeric_params_dct[numeric_cols[count]]
                x_new.append((item - d['mean'])/d['std'])
                count += 1
        return x_new

    def process_data(self, fn):

        print('read data')
        data = self.read_file(fn)

        print('create an iterable')
        if self.sample == 'train':
            if self.baseline_feats:
                for row in data:
                    x1 = row[self.cat_cols].values.tolist()
                    x2 = row[self.numeric_cols].values.tolist()
                    if self.normalize:
                        x2 = self.transform_numeric_cols(
                            self.numeric_params_dct, self.numeric_cols,
                            x2)
                    if self.title_features:
                        item_cols = row[self.item_col].tolist()
                        x2_new = []
                        for i, item in enumerate(item_cols):
                            x2_new.append(
                                x2[i] + self.title_feats_dct[str(item)])
                        x2 = x2_new
                    y = row[self.dv_col].tolist()
                    yield (x1, x2, y)
            else:
                for row in data:
                    user = row[self.user_col].tolist()
                    item = row[self.item_col].tolist()
                    y = row[self.dv_col].tolist()
                    yield (user, item), y
        else:
            if self.baseline_feats:
                for i, row in data.iterrows():
                    x1 = row[self.cat_cols].tolist()
                    x2 = row[self.numeric_cols].tolist()
                    if self.normalize:
                        x2 = self.transform_numeric_cols(
                            self.numeric_params_dct, self.numeric_cols,
                            x2)
                    if self.title_features:
                        item = row[self.item_col]
                        x2 += self.title_feats_dct[str(item)]
                    y = row[self.dv_col]
                    yield (x1, x2, y)
            else:
                for i, row in data.iterrows():
                    yield (row[self.user_col],
                           row[self.item_col]), row[self.dv_col]

    def get_stream(self, files):
        return chain.from_iterable(map(self.process_data, files))

    def __iter__(self):
        return self.get_stream(self.files)

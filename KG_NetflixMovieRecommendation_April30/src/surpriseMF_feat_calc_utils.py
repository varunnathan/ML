import pandas as pd
import time
from surprise import Reader, Dataset, SVD, SVDpp, KNNBaseline, BaselineOnly
from surprise.model_selection import cross_validate


# GLOBALS
NEEDED_COLS = ['User', 'Movie', 'Rating']


class SurpriseMF(object):

    def __init__(self, needed_cols=NEEDED_COLS, rating_scale=(1, 5),
                 algorithm='svd', choose_algorithm_flag=False):
        if algorithm == 'svd':
            self.algo = SVD()
        elif algorithm == 'svdpp':
            self.algo = SVDpp()
        self.choose_algorithm_flag = choose_algorithm_flag
        self.needed_cols = needed_cols
        self.rating_scale = rating_scale
        self.reader = Reader(rating_scale=self.rating_scale)

    def prepare_data_for_training(self, df_train):

        data = Dataset.load_from_df(df_train[self.needed_cols], self.reader)
        train_data = data.build_full_trainset()

        return data, train_data

    def choose_algorithm(self, df_train):

        start = time.time()

        data, _ = self.prepare_data_for_training(df_train)
        del _

        benchmark = []

        print('Iterate over all algorithms')
        for algorithm in [SVD(), SVDpp(), KNNBaseline(), BaselineOnly()]:
            print('Perform cross validation')
            results = cross_validate(algorithm, data, measures=['RMSE'],
                                     cv=3, verbose=False)
            print('time taken: %0.2f' % (time.time() - start))

            print('Get results & append algorithm name')
            tmp = pd.DataFrame.from_dict(results).mean(axis=0)
            tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split(
                '.')[-1]], index=['Algorithm']))
            benchmark.append(tmp)

        print('total time taken: %0.2f' % (time.time() - start))

        out_df = pd.DataFrame(benchmark).set_index('Algorithm').sort_values(
            'test_rmse')

        return out_df

    def fit(self, df_train):

        start = time.time()

        print('data preparation')
        _, train_data = self.prepare_data_for_training(df_train)
        del _
        print('time taken: %0.2f' % (time.time() - start))

        print('training begins')
        self.algo.fit(train_data)
        print('time taken: %0.2f' % (time.time() - start))

        if self.choose_algorithm_flag:
            print('choose the best algorithm based on cross validation')
            out_df = self.choose_algorithm(df_train)
            return out_df

    def predict(self, df_test):

        pred_ratings = []
        count = 0
        for _, row in df_test.iterrows():
            if count % 1000 == 0:
                print('num completed: ', count)
            uid, mid, rui = row['User'], row['Movie'], row['Rating']
            pred = self.algo.predict(uid, mid, r_ui=rui, verbose=False)
            pred_ratings.append(pred.est)
            count += 1

        return pred_ratings

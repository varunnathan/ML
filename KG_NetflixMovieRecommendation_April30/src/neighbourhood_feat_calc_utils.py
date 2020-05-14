import pandas as pd
import numpy as np

# To compute similarities between vectors
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


def _initialize(file_num, user_data_fn, num_user_bins, num_movie_bins):

    print('read overall data')
    inp_fn = user_data_fn.format(file_num)
    print('file: %s' % (inp_fn))
    df = pd.read_hdf(inp_fn, key='stage')
    print('\n')

    print('shape: ', df.shape)
    print('\n')

    print('bins for users and movies based on num_ratings')

    # num_ratings_user
    user_df = df.groupby('User')['Rating'].count().rename(
        'num_rating_user').reset_index()
    user_df['num_rating_user_bins'] = pd.qcut(
        user_df['num_rating_user'], num_user_bins, labels=False)

    # num_ratings_movie
    movie_df = df.groupby('Movie')['Rating'].count().rename(
        'num_rating_movie').reset_index()
    movie_df['num_rating_movie_bins'] = pd.qcut(
        movie_df['num_rating_movie'], num_movie_bins, labels=False)

    # merge
    df = pd.merge(pd.merge(df, user_df, on='User'), movie_df, on='Movie')

    del user_df, movie_df

    return df


def sampling(data, split=0.01, _type='overall'):

    # Shuffle DataFrame
    d = data.sample(frac=1).reset_index(drop=True)

    test_size = int(round(split*d.shape[0]))

    if _type == 'overall':
        # size
        val_size = 4*test_size
        train_size = d.shape[0] - val_size - test_size

        # Split into train & test sets
        df_train = d[:train_size]
        df_val = d[train_size:train_size+val_size]
        df_test = d[train_size+val_size:]

        return df_train, df_val, df_test
    else:
        # Split into train & test sets
        df_test = d[:test_size]
        df_train = d[test_size:]

        return df_train, df_test


def define_segments_for_modelling(df, num_segments=3):

    print('mapping dict')
    user_bins_num_rating_dct = df['num_rating_user_bins'].value_counts().to_dict()
    movie_bins_num_rating_dct = df['num_rating_movie_bins'].value_counts().to_dict()

    print('compute crosstab')
    table = pd.crosstab(df['num_rating_user_bins'],
                        df['num_rating_movie_bins'], margins=True)

    segments = []   # [(user_bin, movie_bin),...]

    print('bins for movies')
    for movie_bin in list(movie_bins_num_rating_dct.keys()):
        user_bins = []
        sparsities = []
        for user_bin in list(user_bins_num_rating_dct.keys()):
            numerator = 100.*table.loc[user_bin, movie_bin]
            denominator = user_bins_num_rating_dct[user_bin]*movie_bins_num_rating_dct[movie_bin]
            sparsity = numerator/denominator
            sparsities.append(sparsity)
            user_bins.append(user_bin)
        idxs = np.argsort(sparsities)[::-1][:num_segments]
        selected_user_bins = [user_bins[idx] for idx in idxs]
        segments += list(zip(selected_user_bins,
                             [movie_bin]*len(selected_user_bins)))

    print('bins for users')
    for user_bin in list(user_bins_num_rating_dct.keys()):
        movie_bins = []
        sparsities = []
        for movie_bin in list(movie_bins_num_rating_dct.keys()):
            numerator = 100.*table.loc[user_bin, movie_bin]
            denominator = user_bins_num_rating_dct[user_bin]*movie_bins_num_rating_dct[movie_bin]
            sparsity = numerator/denominator
            sparsities.append(sparsity)
            movie_bins.append(movie_bin)
        idxs = np.argsort(sparsities)[::-1][:num_segments]
        selected_movie_bins = [movie_bins[idx] for idx in idxs]
        segments += list(zip([user_bin]*len(selected_movie_bins),
                             selected_movie_bins))

    return list(set(segments))


def get_similar_movies_dct(sim_matrix, movie_ids, top_n=20):

    out = {}
    movie_id_mapping_inv = {i: id for i, id in enumerate(movie_ids)}
    for i, movie_id in enumerate(movie_ids):

        if i % 50 == 0:
            print(i)
        # sort similar movies by index
        similar_movie_index = np.argsort(sim_matrix[i])[::-1][:top_n]

        similar_movie_index = [movie_id_mapping_inv[x]
                               for x in similar_movie_index]

        # sort similar movies by score
        similar_movie_score = np.sort(sim_matrix[i])[::-1][:top_n]

        # save
        out[movie_id] = list(zip(similar_movie_index,
                                 similar_movie_score))

    return out


def find_similar_movies(data, num_similar):

    print('Create a user-movie matrix with empty values')
    df_p = data.pivot_table(index='User', columns='Movie',
                            values='Rating')

    print('Shape User-Movie-Matrix:\t{}'.format(df_p.shape))

    print('fill in missing values with mean of each movie')
    df_p_imputed = df_p.fillna(df_p.mean(axis=0)).T

    del df_p

    print(df_p_imputed.shape)

    print('similarity between all users')
    similarity = cosine_similarity(df_p_imputed.values)

    print('remove self-similarity')
    similarity -= np.eye(similarity.shape[0])

    print('shape of similarity matrix: ', similarity.shape)

    movie_ids = df_p_imputed.index.tolist()
    print('num movies: ', len(movie_ids))

    print('get similar movies dct')
    sim_movie_dct = get_similar_movies_dct(similarity, movie_ids,
                                           num_similar)

    return sim_movie_dct


def calc_feature_per_user_movie(user_id, movie_id, sim_movie_dct,
                                user_movies_ratings_dct):

    user_movie_rating_tup = user_movies_ratings_dct.get(user_id, None)
    sim_movie_tup = sim_movie_dct.get(movie_id, None)

    if (user_movie_rating_tup) and (sim_movie_tup):
        sim_movie_ids = [x[0] for x in sim_movie_tup]
        sim_movie_scores = [x[1] for x in sim_movie_tup]
        numerator, denominator = 0, 0
        for movie_id, rating in user_movie_rating_tup:
            if movie_id in sim_movie_ids:
                idx = sim_movie_ids.index(movie_id)
                score = sim_movie_scores[idx]
                numerator += (rating*score)
                denominator += score

        return numerator/denominator if denominator > 0 else None


def calc_features_test_df(test_df, sim_movie_dct,
                          user_movies_ratings_dct, user_col='User',
                          movie_col='Movie'):

    scores = []
    count = 1
    for _, row in test_df.iterrows():
        if count % 1000 == 0:
            print(count)
        user_id, movie_id = row[user_col], row[movie_col]
        score = calc_feature_per_user_movie(
            str(user_id), str(movie_id), sim_movie_dct, user_movies_ratings_dct)
        scores.append(score)
        count += 1

    return scores


def calc_rmse(pred_df, target_col='Rating',
              score_col='predicted_rating'):
    mask = pred_df[score_col].notnull()
    if mask.sum():
        y_true = pred_df.loc[mask, target_col]
        y_pred = pred_df.loc[mask, score_col]
        rmse = np.sqrt(mean_squared_error(y_true=y_true, y_pred=y_pred))
        return rmse
    else:
        return None


def segment_modelling(df_train, user_bin, movie_bin, split=0.1,
                      num_similar=150):

    print('define segment')
    mask1 = df_train['num_rating_user_bins'] == user_bin
    mask2 = df_train['num_rating_movie_bins'] == movie_bin

    df_train_segment = df_train.loc[mask1&mask2, :]
    df_train_segment.reset_index(drop=True, inplace=True)

    print('overall shape: ', df_train_segment.shape)

    print('sampling')
    df_train_segment_train, df_train_segment_test = sampling(
        df_train_segment, split=split, _type='segment')

    del df_train_segment

    print('train shape: ', df_train_segment_train.shape)
    print('test shape: ', df_train_segment_test.shape)

    print('find similar movies')
    sim_movie_dct = find_similar_movies(df_train_segment_train,
                                        num_similar)

    print('find movies rated by users and their corresponding ratings')
    df_train_segment_train['movie_rating_tup'] = list(
        map(lambda x, y: (x, y), df_train_segment_train['Movie'],
            df_train_segment_train['Rating']))
    user_movies_ratings_df = df_train_segment_train.groupby('User')[
        'movie_rating_tup'].apply(list).rename(
        'movie_rating_tup').reset_index()
    user_movies_ratings_dct = dict(
        zip(user_movies_ratings_df['User'],
            user_movies_ratings_df['movie_rating_tup']))
    del user_movies_ratings_df

    print('Evaluation on test set')
    scores = calc_features_test_df(df_train_segment_test, sim_movie_dct,
                                   user_movies_ratings_dct)

    df_train_segment_test['predicted_rating'] = scores

    rmse = calc_rmse(df_train_segment_test, target_col='Rating',
                     score_col='predicted_rating')

    return (sim_movie_dct, user_movies_ratings_dct,
            df_train_segment_train, df_train_segment_test, rmse)

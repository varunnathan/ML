import numpy as np
import copy, json


def get_weighted_mean_ratings(df, m, group_col='Movie',
                              agg_col='Rating'):

    # overall mean of all movies
    C = df[agg_col].mean()

    # mean by movies
    R = df.groupby(group_col)[agg_col].mean()

    # count by movies
    v = df.groupby(group_col)[agg_col].count().values

    # movie indices
    movie_ids = R.index
    R = R.values

    # weighted score calculation
    weighted_scores = (v/(v+m))*R + (m/(v+m))*C

    # rank based on weighted score
    weighted_ranking = np.argsort(weighted_scores)[::-1]
    weighted_scores = np.sort(weighted_scores)[::-1]

    # get movie ids corresponding to rankings
    weighted_movie_ids = movie_ids[weighted_ranking]

    weighted_scores_dct = dict(zip(weighted_movie_ids, weighted_scores))

    return weighted_scores_dct


def _update_user_dct(file_dct, global_dct, file_dct_count=None,
                     global_dct_count=None, how='earliest'):

    if not global_dct:
        return file_dct

    d = copy.deepcopy(global_dct)
    if how == 'earliest':
        func = lambda x, y: min(x, y)
    elif how == 'latest':
        func = lambda x, y: max(x, y)
    elif how == 'count':
        func = lambda x, y: sum([x, y])

    for k in file_dct:
        if k in global_dct:
            if how == 'mean':
                numerator = sum([(file_dct[k] * file_dct_count[k]),
                                 (global_dct[k] * global_dct_count[k])])
                denominator = sum([file_dct_count[k], global_dct_count[k]])
                d[k] = 1.*numerator/denominator
            else:
                d[k] = func(file_dct[k], global_dct[k])
        else:
            d[k] = file_dct[k]

    return d


def init_feat_dict(fn):
    if os.path.isfile(fn):
        return json.load(open(fn))
    else:
        return {}


def baseline_feat_calc_helper(df, group_col, func_type):
    """
    func_type: mean, count, earliest, latest
    group_col: User, Movie
    """
    if func_type in ('mean', 'count'):
        agg_col = 'Rating'
    elif func_type in ('earliest', 'latest'):
        agg_col = 'Date'

    agg_name = '_'.join([func_type, agg_col])

    func_dict = {'mean': lambda x: x.mean(),
                 'count': lambda x: x.count(),
                 'earliest': lambda x: x.min(),
                 'latest': lambda x: x.max()}

    grouped_df = func_dict[func_type](
        df.groupby(group_col)[agg_col]).rename(agg_name).reset_index()

    if func_type in ('earliest', 'latest'):
        grouped_df[agg_name] = grouped_df[agg_name].apply(
            lambda x: x.strftime('%Y-%m-%d'))

    d = dict(zip(grouped_df[group_col], grouped_df[agg_name]))

    return d

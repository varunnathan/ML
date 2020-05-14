import pandas as pd
from collections import deque


def load_single_user_file(user_file_num):
    """
    loads a single user file into memory
    """
    print('read user data')
    df = pd.read_csv(USER_DATA_FN.format(user_file_num), header=None,
                     names=['User', 'Rating', 'Date'],
                     usecols=[0, 1, 2])
    print('Shape user-data:\t{}'.format(df.shape))

    print('convert Date to datetime format')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    print('Find empty rows to slice dataframe for each movie')
    tmp_movies = df[df['Rating'].isnull()]['User'].reset_index()
    movie_indices = [[index, int(movie[:-1])] for index, movie in
                     tmp_movies.values]

    print('Shift the movie_indices by one to get start and endpoints of all movies')
    shifted_movie_indices = deque(movie_indices)
    shifted_movie_indices.rotate(-1)

    print('create a dataframe with movie id and user ratings')
    user_data = []

    for [df_id_1, m_id], [df_id_2, n_m_id] in zip(movie_indices,
                                                  shifted_movie_indices):
        if df_id_1 < df_id_2:
            tmp_df = df.loc[df_id_1+1: df_id_2-1, :].copy()
        else:
            # last movie
            tmp_df = df.loc[df_id_1+1:, :].copy()

        tmp_df['Movie'] = m_id
        user_data.append(tmp_df)

    print('Combine all dataframes')
    df_1 = pd.concat(user_data)
    del (user_data, df, tmp_movies, tmp_df, shifted_movie_indices,
         movie_indices, df_id_1, m_id, df_id_2, n_m_id)
    print('Shape User-Ratings:\t{}'.format(df_1.shape))
    print('num users: ', df_1['User'].nunique())

    return df_1


def load_qualifying_data(inp_fn):
    """
    loads a single user file into memory
    """
    print('read user data')
    df = pd.read_csv(inp_fn, header=None,
                     names=['User', 'Date'],
                     usecols=[0, 1])
    print('Shape user-data:\t{}'.format(df.shape))

    print('convert Date to datetime format')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

    print('Find empty rows to slice dataframe for each movie')
    tmp_movies = df[df['Date'].isnull()]['User'].reset_index()
    movie_indices = [[index, int(movie[:-1])] for index, movie in
                     tmp_movies.values]

    print('Shift the movie_indices by one to get start and endpoints of all movies')
    shifted_movie_indices = deque(movie_indices)
    shifted_movie_indices.rotate(-1)

    print('create a dataframe with movie id and user ratings')
    user_data = []

    for [df_id_1, m_id], [df_id_2, n_m_id] in zip(movie_indices,
                                                  shifted_movie_indices):
        if df_id_1 < df_id_2:
            tmp_df = df.loc[df_id_1+1: df_id_2-1, :].copy()
        else:
            # last movie
            tmp_df = df.loc[df_id_1+1:, :].copy()

        tmp_df['Movie'] = m_id
        user_data.append(tmp_df)

    print('Combine all dataframes')
    df_1 = pd.concat(user_data)
    del (user_data, df, tmp_movies, tmp_df, shifted_movie_indices,
         movie_indices, df_id_1, m_id, df_id_2, n_m_id)
    print('Shape User-Ratings:\t{}'.format(df_1.shape))
    print('num users: ', df_1['User'].nunique())

    return df_1

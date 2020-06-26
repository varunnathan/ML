import os

LOCAL_DIR = '/Users/varunn/Documents/'
PROJ_DIR = os.path.join(LOCAL_DIR, 'kaggle')
DATA_DIR = os.path.join(PROJ_DIR, 'netflix-prize-data')
OUT_DIR = os.path.join(DATA_DIR, 'interim')
METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
MOVIE_METADATA_DIR = os.path.join(DATA_DIR, 'movie_metadata')
PREPARED_DATA_DIR = os.path.join(DATA_DIR, 'prepared_data_for_NN_modelling')
FEATS_DIR = os.path.join(DATA_DIR, 'features')
USER_DATA_FN = os.path.join(DATA_DIR, 'combined_data_{}.txt')
MOVIES_DATA_FN = os.path.join(DATA_DIR, 'movie_titles.csv')
PROBE_DATA_FN = os.path.join(DATA_DIR, 'probe.txt')
QUALIFYING_DATA_FN = os.path.join(DATA_DIR, 'qualifying.txt')
SAMPLE_DATA_DIR = os.path.join(DATA_DIR, 'interim_sampled')
SAMPLE_FEATS_DIR = os.path.join(DATA_DIR, 'features_sampled')
BASELINE_FEATS_DIR = os.path.join(DATA_DIR, 'baseline_features')
VISUALIZATION_DIR = os.path.join(DATA_DIR, 'visualization')
NEIGHBOURHOOD_FEATS_DIR = os.path.join(DATA_DIR, 'neighbourhood_features')
SURPRISEMF_FEATS_DIR = os.path.join(DATA_DIR, 'surprisemf_features')
SURPRISEMF_MODEL_OBJ_FN = os.path.join(SURPRISEMF_FEATS_DIR,
                                       'fitted_svd_model_U{}_M{}_{}.pkl')
SURPRISEMF_CLASS_FN = os.path.join(SURPRISEMF_FEATS_DIR,
                                   'fitted_svd_class_U{}_M{}_{}.pkl')
SURPRISEMF_TEST_RMSE_FN = os.path.join(SURPRISEMF_FEATS_DIR, 'segment_test_rmse_{}.csv')
SURPRISEMF_SEGMENT_TEST_FN = os.path.join(SURPRISEMF_FEATS_DIR,
                                          'segment_test_df_U{}_M{}_{}.csv')
LIGHTFM_FEATS_DIR = os.path.join(DATA_DIR, 'lightfm_features')
LightFM_MODEL_FN = os.path.join(LIGHTFM_FEATS_DIR, 'lightfm_model_{}.pkl')
LIGHTFM_INTERACTION_FN = os.path.join(LIGHTFM_FEATS_DIR, 'lightfm_interaction_{}.pkl')
LIGHTFM_USER_2_INDEX_MAP_FN = os.path.join(LIGHTFM_FEATS_DIR,
                                           'lightfm_user2index_dct_{}.json')
LIGHTFM_ITEM_2_INDEX_MAP_FN = os.path.join(LIGHTFM_FEATS_DIR,
                                           'lightfm_item2index_dct_{}.json')
INP_FN = os.path.join(OUT_DIR, 'user_data_{}.h5')
TRAIN_FN = os.path.join(OUT_DIR, 'user_train_data_{}.h5')
TEST_FN = os.path.join(OUT_DIR, 'user_test_data_{}.h5')
VAL_FN = os.path.join(OUT_DIR, 'user_val_data_{}.h5')
USER2IDX_FN = os.path.join(METADATA_DIR, 'user2idx.json')
IDX2USER_FN = os.path.join(METADATA_DIR, 'idx2user.json')
ITEM2IDX_FN = os.path.join(METADATA_DIR, 'item2idx.json')
IDX2ITEM_FN = os.path.join(METADATA_DIR, 'idx2item.json')
FEATS_TEST_FN = os.path.join(SAMPLE_FEATS_DIR, 'features_data_test_{}.h5')
FEATS_VAL_FN = os.path.join(SAMPLE_FEATS_DIR, 'features_data_val_{}.h5')
USER_MOVIE_RATINGS_DCT_FN = os.path.join(NEIGHBOURHOOD_FEATS_DIR,
                                         'user_movie_ratings_dct_{}.json')
SIM_MOVIE_DCT_FN = os.path.join(NEIGHBOURHOOD_FEATS_DIR,
                                'sim_movie_dct_U{}_M{}_{}.json')
SIM_MOVIE_DCT_GLOBAL_FN = os.path.join(NEIGHBOURHOOD_FEATS_DIR,
                                       'sim_movie_dct_global_{}.json')
MOVIE_SEGMENT_DCT_FN = os.path.join(NEIGHBOURHOOD_FEATS_DIR,
                                    'movie_segment_dct_{}.json')
SEGMENT_TRAIN_FN = os.path.join(NEIGHBOURHOOD_FEATS_DIR,
                                'segment_train_df_U{}_M{}_{}.csv')
SEGMENT_TEST_FN = os.path.join(NEIGHBOURHOOD_FEATS_DIR,
                               'segment_test_df_U{}_M{}_{}.csv')
TEST_RMSE_FN = os.path.join(NEIGHBOURHOOD_FEATS_DIR,
                            'segment_test_rmse_{}.csv')
MEAN_RATINGS_MOVIE_DCT_FN = os.path.join(BASELINE_FEATS_DIR,
                                         'mean_ratings_movie_dct.json')
MEAN_RATINGS_USER_DCT_FN = os.path.join(BASELINE_FEATS_DIR,
                                        'mean_ratings_user_dct.json')
NUM_RATINGS_MOVIE_DCT_FN = os.path.join(BASELINE_FEATS_DIR,
                                        'num_ratings_movie_dct.json')
NUM_RATINGS_USER_DCT_FN = os.path.join(BASELINE_FEATS_DIR,
                                       'num_ratings_user_dct.json')
WEIGHTED_MEAN_RATINGS_MOVIE_DCT_FN = os.path.join(
    BASELINE_FEATS_DIR, 'weighted_mean_ratings_movie_dct.json')
WEIGHTED_MEAN_RATINGS_USER_DCT_FN = os.path.join(
    BASELINE_FEATS_DIR, 'weighted_mean_ratings_user_dct.json')
EARLIEST_RATING_DATE_MOVIE_FN = os.path.join(
    BASELINE_FEATS_DIR, 'earliest_rating_date_movie_dct.json')
EARLIEST_RATING_DATE_USER_FN = os.path.join(
    BASELINE_FEATS_DIR, 'earliest_rating_date_user_dct.json')
LATEST_RATING_DATE_MOVIE_FN = os.path.join(
    BASELINE_FEATS_DIR, 'latest_rating_date_movie_dct.json')
LATEST_RATING_DATE_USER_FN = os.path.join(
    BASELINE_FEATS_DIR, 'latest_rating_date_user_dct.json')
FILE_NUMS = range(1, 5, 1)
MODEL_DIR = os.path.join(DATA_DIR, 'models')
PREDICTION_DIR = os.path.join(DATA_DIR, 'predictions')
REG_MODEL_IMPUTE_DCT_FN = os.path.join(MODEL_DIR,
                                       'reg_model_impute_dct_{}.json')
REG_MODEL_OBJ_FN = os.path.join(MODEL_DIR, 'reg_model_obj_{}.pkl')
REG_MODEL_PRED_VAL_FN = os.path.join(PREDICTION_DIR,
                                     'reg_model_prediction_val_{}.h5')
REG_MODEL_PRED_TEST_FN = os.path.join(PREDICTION_DIR,
                                      'reg_model_prediction_test_{}.h5')
NUMERIC_FEATS_PARAMS_DCT_FN = os.path.join(METADATA_DIR,
                                           'numeric_feats_params_dct.json')
MOVIE_TITLES_TFIDF_COMPS_FN = os.path.join(MOVIE_METADATA_DIR,
                                           'movie_titles_tfidf_comps.json')
MOVIE_TITLES_TFIDF_FEAT_IMP_FN = os.path.join(MOVIE_METADATA_DIR,
                                              'movie_titles_tfidf_feat_imp.csv')
MOVIE_TITLES_TFIDF_PIPELINE_FN = os.path.join(MOVIE_METADATA_DIR,
                                              'movie_titles_tfidf_pipeline.pkl')
MOVIE_TITLES_BERT_COMPS_FN = os.path.join(MOVIE_METADATA_DIR,
                                          'movie_titles_bert_comps.json')

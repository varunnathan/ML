import numpy as np
import time
from scipy.sparse import coo_matrix
from scipy import sparse
from lightfm import LightFM
from lightfm.evaluation import auc_score, precision_at_k, recall_at_k


def get_user_list(df, user_column):
    """

    creating a list of user from dataframe df, user_column is a column
    consisting of users in the dataframe df

    """

    return np.sort(df[user_column].unique())

def get_item_list(df, item_name_column):

    """

    creating a list of items from dataframe df, item_column is a column
    consisting of items in the dataframe df

    return to item_id_list and item_id2name_mapping

    """

    item_list = df[item_name_column].unique()


    return item_list

# creating user_id, item_id, and features_id

def id_mappings(user_list, item_list):
    """

    Create id mappings to convert user_id, item_id, and feature_id

    """
    user_to_index_mapping = {}
    index_to_user_mapping = {}
    for user_index, user_id in enumerate(user_list):
        user_to_index_mapping[user_id] = user_index
        index_to_user_mapping[user_index] = user_id

    item_to_index_mapping = {}
    index_to_item_mapping = {}
    for item_index, item_id in enumerate(item_list):
        item_to_index_mapping[item_id] = item_index
        index_to_item_mapping[item_index] = item_id


    return user_to_index_mapping, index_to_user_mapping, \
           item_to_index_mapping, index_to_item_mapping


def get_interaction_matrix(df, df_column_as_row, df_column_as_col,
                           df_column_as_value, row_indexing_map,
                           col_indexing_map):

    row = df[df_column_as_row].apply(
        lambda x: row_indexing_map[x]).values
    col = df[df_column_as_col].apply(
        lambda x: col_indexing_map[x]).values
    value = df[df_column_as_value].values

    return coo_matrix((value, (row, col)),
                      shape = (len(row_indexing_map),
                               len(col_indexing_map)))


def lightfm_train_utils(user_to_product_interaction_train, **kwargs):

    # get params
    no_components = kwargs['no_components']
    loss = kwargs['loss']
    epochs = kwargs['epochs']
    model_out_fn = kwargs['model_out_fn']

    # initialising model with warp loss function
    model = LightFM(no_components = no_components, loss = loss)

    # fitting into user to product interaction matrix only / pure collaborative filtering factor
    start = time.time()
    #===================

    model.fit(user_to_product_interaction_train,
              user_features=None,
              item_features=None,
              sample_weight=None,
              epochs=epochs,
              num_threads=8,
              verbose=False)

    #===================
    end = time.time()
    print("time taken = {0:.{1}f} seconds".format(end - start, 2))

    # save
    joblib.dump(model, open(model_out_fn, 'wb'))

    return model


def lightfm_eval_utils(user_to_product_interaction_test, model, **kwargs):

    eval_k = kwargs['eval_k']

    # test

    start = time.time()

    auc_without_features = auc_score(
        model = model,
        test_interactions = user_to_product_interaction_test,
        num_threads = 8, check_intersections = False)

    print("average AUC without adding item-feature interaction = {0:.{1}f}".format(
        auc_without_features.mean(), 2))

    prec_at_k = precision_at_k(model,
                               user_to_product_interaction_test,
                               k=eval_k).mean()
    print('Test precision at k={}:\t\t{:.4f}'.format(eval_k, prec_at_k))

    rec_at_k = recall_at_k(model,
                           user_to_product_interaction_test,
                           k=eval_k).mean()
    print('Test Recall at k={}:\t\t{:.4f}'.format(eval_k, rec_at_k))

    print('time taken: %0.4f' % (time.time() - start))


class recommendation_sampling:

    def __init__(self, model, items, user_to_product_interaction_matrix,
                 user2index_map, item2index_map):

        self.user_to_product_interaction_matrix = user_to_product_interaction_matrix
        self.model = model
        self.items = items
        self.user2index_map = user2index_map
        self.item2index_map = item2index_map

    def recommendation_for_user(self, user):

        # getting the userindex

        userindex = self.user2index_map.get(user, None)

        if userindex == None:
            return None

        users = [userindex]

        # products already bought

        known_positives = self.items[self.user_to_product_interaction_matrix.tocsr()[userindex].indices]

        # scores from model prediction
        scores = self.model.predict(user_ids = users, item_ids = np.arange(self.user_to_product_interaction_matrix.shape[1]))

        #print(scores)
        #print(list(zip(self.items, scores)))
        # top items

        top_items = self.items[np.argsort(-scores)]

        # printing out the result
        print("User %s" % user)
        print("     Known positives:")

        for x in known_positives[:3]:
            print("                  %s" % x)


        print("     Recommended:")

        for x in top_items[:3]:
            print("                  %s" % x)

    def get_score(self, user_lst, movie_lst):

        user_id_lst = [self.user2index_map[str(x)] for x in user_lst]
        item_id_lst = [self.item2index_map[x] for x in movie_lst]
        scores = self.model.predict(user_ids=np.array(user_id_lst),
                                    item_ids=np.array(item_id_lst))
        return scores

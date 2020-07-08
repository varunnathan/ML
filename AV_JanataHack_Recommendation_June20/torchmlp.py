import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.data as data


class PairwiseInteractionsMLP(data.Dataset):
    """
    Sample data from an interactions matrix in a pairwise fashion. The row is
    treated as the main dimension, and the columns are sampled pairwise.
    """

    def __init__(self, data, n_items, n_challenges_per_user,
                 use_item_attr, seen_items_dct_all, item_attr_dct,
                 seed=1):
        self.df = data
        self.seen_items_dct_all = seen_items_dct_all
        self.item_attr_dct = item_attr_dct
        self.seed = seed
        self.n_items = n_items
        self.n_challenges_per_user = n_challenges_per_user
        self.use_item_attr = use_item_attr
        self.user_cols = ['user_id']
        self.item_cols = ['programming_language', 'challenge_series_ID',
                          'author_ID', 'author_gender',
                          'author_org_ID', 'category_id']
        self.joint_cols = ['challenge_sequence']
        if self.use_item_attr:
            self.cat_cols = self.user_cols + ['challenge'] + self.item_cols + self.joint_cols
        else:
            self.cat_cols = self.user_cols + ['challenge']
        self.num_cols = ['total_submissions']

    def __getitem__(self, index):
        row = self.df.iloc[index]
        user_id = row['user_id']
        user_values = [user_id]

        pos_cat_values = np.array([row[col] for col in self.cat_cols])
        pos_num_values = np.array([row[col] for col in self.num_cols])

        np.random.seed(self.seed)
        neg_item = np.random.permutation(
            [x for x in range(self.n_items) if x not in
             self.seen_items_dct_all[user_id]])[0]
        neg_item_values = [self.item_attr_dct[neg_item][i] for i, col in
                           enumerate(self.item_cols)]
        neg_num_values = np.array([self.item_attr_dct[neg_item][6]])

        if self.use_item_attr:
            neg_joint_values = [self.n_challenges_per_user]
            neg_cat_values = np.array(user_values + [neg_item] + neg_item_values +
                                      neg_joint_values)
        else:
            neg_cat_values = np.array(user_values + [neg_item])

        return pos_cat_values, pos_num_values, neg_cat_values, neg_num_values

    def __len__(self):
        return self.df.shape[0]


class BPRModuleMLP(nn.Module):
    """
    Defines the neural network for challenge recommendation
    """

    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for
                                         categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 1)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.emb_drop = nn.Dropout(0.6)

    def calc_pred(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = self.lin1(x)
        return x

    def forward(self, pos_cat, pos_num, neg_cat, neg_num):
        pos_preds = self.calc_pred(pos_cat, pos_num)
        neg_preds = self.calc_pred(neg_cat, neg_num)
        return pos_preds - neg_preds

import torch
from torch import nn, tensor
import numpy as np
import collections

from abc import ABCMeta
from abc import abstractmethod
from typing import Callable
from tqdm import tqdm

from metrics import mapk

from training import Step
from torchmf import bpr_loss


class StepMLP(Step):
    """Batch training of MLP algorithm for recommendation."""
    def __init__(self, model: torch.nn.Module, n_items, item_attr_dct,
                 n_challenges_per_user, use_item_attr, actual_items_dct,
                 seen_items_dct, loss_function=bpr_loss,
                 optimizer = torch.optim.Adam,
                 lr = 0.01, weight_decay = 0., batch_size=512,
                 num_predictions=3):
        super(StepMLP, self).__init__(
            model, actual_items_dct, seen_items_dct, loss_function, optimizer,
            lr, weight_decay, batch_size, num_predictions)
        self.model = self.model.double()
        self.n_items = n_items
        self.item_attr_dct = item_attr_dct
        self.n_challenges_per_user = n_challenges_per_user
        self.use_item_attr = use_item_attr
        self.item_cols = ['programming_language', 'challenge_series_ID',
                          'author_ID', 'author_gender',
                          'author_org_ID', 'category_id']

    def _training(self, data_loader: torch.utils.data.DataLoader,
                  data_size: int):
        """Trains the model on a batch of user-item interactions."""

        self.model.train()
        total_loss = torch.Tensor([0])
        with tqdm(total=data_size//self.batch_size) as pbar:
            for pos_cat, pos_num, neg_cat, neg_num in data_loader:
                self.optimizer.zero_grad()

                pos_cat = pos_cat.long()
                neg_cat = neg_cat.long()
                pos_num = pos_num.double()
                neg_num = neg_num.double()

                preds = self.model(pos_cat, pos_num, neg_cat, neg_num)
                loss = self.loss_function(preds)
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()
                batch_loss = loss.item() / pos_cat.size()[0]

                pbar.update(1)

        total_loss /= data_size
        return total_loss

    def _validation(self, data_loader: torch.utils.data.DataLoader,
                    data_size: int, calc_mapk: bool):
        """Evaluates the model on a batch of user-item interactions"""

        self.model.eval()
        total_loss = torch.Tensor([0])
        if calc_mapk:
            users_processed = set()

        with tqdm(total=data_size//self.batch_size) as pbar:
            for pos_cat, pos_num, neg_cat, neg_num in data_loader:

                pos_cat = pos_cat.long()
                neg_cat = neg_cat.long()
                pos_num = pos_num.double()
                neg_num = neg_num.double()

                pred = self.model(pos_cat, pos_num, neg_cat, neg_num)
                loss = self.loss_function(pred)
                total_loss += loss.item()

                if calc_mapk:
                    user_lst_batch = set(pos_cat.numpy()[:, 0]) - users_processed
                    users_processed.update(user_lst_batch)
                    recommended_items = self.recommend_batch(user_lst_batch)
                    d = dict(zip(user_lst_batch, recommended_items))
                    self.recommended_items_dct.update(d)

                pbar.update(1)

        total_loss /= data_size
        if calc_mapk:
            test_mapk = self.get_mapk()
            return total_loss[0], test_mapk
        else:
            return total_loss[0]

    def recommend(self, user, k:int = 10):
        """Recommends the top-k items to a specific user."""
        self.model.eval()

        item_lst = [x for x in range(self.n_items) if x not in
                    self.seen_items_dct[user]]
        user_value = [user]
        cat_values, num_values = [], []
        for item in item_lst:
            item_value = [self.item_attr_dct[item][i] for i, col in
                          enumerate(self.item_cols)]
            num_value = [self.item_attr_dct[item][6]]
            if self.use_item_attr:
                joint_value = [self.n_challenges_per_user]
                cat_value = user_value + [item] + item_value + joint_value
            else:
                cat_value = user_value + [item]
            cat_values.append(cat_value)
            num_values.append(num_value)

        cat_values = tensor(cat_values)
        cat_values = cat_values.long()
        num_values = tensor(num_values)
        num_values = num_values.double()
        scores = self.model.calc_pred(cat_values, num_values)
        scores = scores.squeeze()
        sorted_scores = scores.argsort().tolist()
        return sorted_scores[::-1][:k]

    def recommend_batch(self, user_list):
        return [self.recommend(user, self.num_predictions) for user in
                user_list]

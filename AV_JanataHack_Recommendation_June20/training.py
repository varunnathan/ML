import torch
from torch import nn, tensor
import numpy as np
import collections

from abc import ABCMeta
from abc import abstractmethod
from typing import Callable
from tqdm import tqdm

from metrics import mapk


class StepBase:
    """Defines the interface that all step models here expose."""
    __metaclass__ = ABCMeta

    @abstractmethod
    def batch_fit(self, data_loader: torch.utils.data.DataLoader, epochs: int):
        """Trains the model on a batch of user-item interactions."""
        pass

    @abstractmethod
    def step(self, user: torch.tensor, item: torch.tensor,
             rating: torch.tensor, preference: torch.tensor):
        """Trains the model incrementally."""
        pass

    @abstractmethod
    def predict(self, user: torch.tensor, k: int):
        """Recommends the top-k items to a specific user."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Saves the model parameters to the given path."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Loads the model parameters from a given path."""
        pass


class Step(StepBase):
    """Incremental and batch training of MF algorithm for recommendation."""
    def __init__(self, model: torch.nn.Module,
                 actual_items_dct, seen_items_dct,
                 loss_function=torch.nn.MSELoss(reduction='sum'),
                 optimizer = torch.optim.Adam,
                 lr = 0.01, weight_decay = 0., batch_size=512,
                 num_predictions=3):
        self.model = model
        self.loss_function = loss_function
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.optimizer = optimizer(self.model.parameters(),
                                   lr=self.lr,
                                   weight_decay=self.weight_decay)
        self.metrics = []
        self.num_predictions = num_predictions
        self.actual_items_dct = actual_items_dct
        self.seen_items_dct = seen_items_dct
        self.recommended_items_dct = {}

    def batch_fit(self, train_loader: torch.utils.data.DataLoader,
                  test_loader: torch.utils.data.DataLoader,
                  train_size: int, test_size: int, epochs: int = 1,
                  calc_mapk: bool = True):
        """Trains the model on a batch of user-item interactions."""

        for epoch in range(epochs):
            stats = {'epoch': epoch+1}

            print('Training begins...')
            train_loss = self._training(train_loader, train_size)
            stats['train_loss'] = train_loss

            print('Validation begins...')
            if calc_mapk:
                print('validation with mapk')
                val_loss, val_mapk = self._validation(
                    test_loader, test_size, calc_mapk)
                stats['val_mapk'] = val_mapk
            else:
                print('validation without mapk')
                val_loss = self._validation(
                    test_loader, test_size, calc_mapk)
            stats['val_loss'] = val_loss
            print(stats)

            self.metrics.append(stats)

    def _training(self, data_loader: torch.utils.data.DataLoader,
                  data_size: int):
        """Trains the model on a batch of user-item interactions."""

        self.model.train()
        total_loss = torch.Tensor([0])
        with tqdm(total=data_size//self.batch_size) as pbar:
            for _, ((row, col), val) in enumerate(data_loader):
                self.optimizer.zero_grad()

                row = row.long()
                if isinstance(col, list):
                    col = tuple(c.long() for c in col)
                else:
                    col = col.long()

                preds = self.model(row, col)
                loss = self.loss_function(preds)
                loss.backward()

                self.optimizer.step()

                total_loss += loss.item()
                batch_loss = loss.item() / row.size()[0]

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
            for _, ((row, col), val) in enumerate(data_loader):

                row = row.long()
                if isinstance(col, list):
                    col = tuple(c.long() for c in col)
                else:
                    col = col.long()

                pred = self.model(row, col)
                loss = self.loss_function(pred)
                total_loss += loss.item()

                if calc_mapk:
                    user_lst_batch = set(row.tolist()) - users_processed
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

    def step(self, user: torch.tensor, item: torch.tensor):
        """Trains the model incrementally."""
        self.model.train()

        self.optimizer.zero_grad()

        pred = self.model(user, item)
        loss = self.loss_function(pred)
        loss.backward()

        self.optimizer.step()

        batch_loss = loss.item()
        return batch_loss

    def recommend(self, user: torch.tensor, k:int = 10) -> torch.tensor:
        """Recommends the top-k items to a specific user."""
        self.model.eval()

        u_embed_one = self.model.pred_model.user_embeddings(user)
        u_embed_one_reshaped = u_embed_one.reshape((
            1, u_embed_one.shape[0]))
        m_embed = self.model.pred_model.item_embeddings.weight
        u_bias_one = self.model.pred_model.user_biases(user)
        u_bias_one_reshaped = u_bias_one.reshape((
            1, u_bias_one.shape[0]))
        m_bias = self.model.pred_model.item_biases.weight

        bias_sum = u_bias_one_reshaped + m_bias
        bias_sum = bias_sum.reshape((bias_sum.shape[1],
                                     bias_sum.shape[0]))

        preds = torch.matmul(u_embed_one_reshaped, m_embed.t())+bias_sum

        sorted_preds = preds.squeeze().argsort().tolist()
        items_seen = self.seen_items_dct[user.item()]
        sorted_preds = [x for x in sorted_preds if x not in items_seen]
        return sorted_preds[::-1][:k]

    def recommend_batch(self, user_list):
        return [self.recommend(tensor(user), self.num_predictions) for user in
                user_list]

    def get_mapk(self):

        actuals, preds = [], []
        for user in self.actual_items_dct:
            actual_item_lst = self.actual_items_dct[user]
            pred_item_lst = self.recommended_items_dct[user]
            actuals.append(actual_item_lst)
            preds.append(pred_item_lst)

        return mapk(actuals, preds, k=self.num_predictions)

    def save(self, path: str):
        """Saves the model parameters to the given path."""
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """Loads the model parameters from a given path."""
        self.model.load_state_dict(torch.load(path))

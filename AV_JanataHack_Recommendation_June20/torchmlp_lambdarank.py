import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.data as data


class DataPreparation(object):
    def __init__(self, dev_df, val_df, item_attr_dct, )


class RankDataset(data.Dataset):



class LambdaRankMLP(nn.Module):
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

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = self.lin1(x)
        return x

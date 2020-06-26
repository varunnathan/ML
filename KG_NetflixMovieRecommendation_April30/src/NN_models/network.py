import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, factors: int = 16,
                 user_embeddings: torch.tensor = None,
                 freeze_users: bool = False,
                 item_embeddings: torch.tensor = None,
                 freeze_items: bool = False,
                 init: nn.init = nn.init.normal_,
                 binary: bool =False, **kwargs):
        super().__init__()
        self.binary = binary

        self.user_embeddings = self._create_embedding(
            n_users, factors, user_embeddings, freeze_users,
            init, **kwargs)
        self.item_embeddings = self._create_embedding(
            n_items, factors, item_embeddings, freeze_items,
            init, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u: torch.tensor, i: torch.tensor) -> torch.tensor:
        user_embedding = self.user_embeddings(u)
        user_embedding = user_embedding[:, None, :]
        item_embedding = self.item_embeddings(i)
        item_embedding = item_embedding[:, None, :]
        rating = torch.matmul(user_embedding, item_embedding.transpose(
            1, 2))
        if self.binary:
            return self.sigmoid(rating)
        return rating

    def _create_embedding(self, n_items, factors, weights, freeze,
                          init, **kwargs):
        embedding = nn.Embedding(n_items, factors)
        init(embedding.weight.data, **kwargs)

        if weights is not None:
            embedding.load_state_dict({'weight': weights})
        if freeze:
            embedding.weight.requires_grad = False

        return embedding


class SimpleCFWithBias(nn.Module):
    """
    Base module for explicit matrix factorization.
    """

    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=40,
                 dropout_p=0,
                 sparse=False,
                 user_embeddings: torch.tensor = None,
                 user_biases: torch.tensor = None,
                 freeze_users: bool = False,
                 item_embeddings: torch.tensor = None,
                 item_biases: torch.tensor = None,
                 freeze_items: bool = False,
                 init: nn.init = nn.init.normal_,
                 **kwargs):
        """
        Parameters
        ----------
        n_users : int
            Number of users
        n_items : int
            Number of items
        n_factors : int
            Number of latent factors (or embeddings or whatever you want to
            call it).
        dropout_p : float
            p in nn.Dropout module. Probability of dropout.
        sparse : bool
            Whether or not to treat embeddings as sparse. NOTE: cannot use
            weight decay on the optimizer if sparse=True. Also, can only use
            Adagrad.
        """
        super(BaseModule, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.user_embeddings, self.user_biases = self._create_embedding(
            n_users, n_factors, user_embeddings, user_biases,
            freeze_users, init, sparse, **kwargs)
        self.item_embeddings, self.item_biases = self._create_embedding(
            n_items, n_factors, item_embeddings, item_biases,
            freeze_items, init, sparse, **kwargs)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(p=self.dropout_p)

        self.sparse = sparse

    def forward(self, users, items):
        """
        Forward pass through the model. For a single user and item, this
        looks like:
        user_bias + item_bias + user_embeddings.dot(item_embeddings)
        Parameters
        ----------
        users : np.ndarray
            Array of user indices
        items : np.ndarray
            Array of item indices
        Returns
        -------
        preds : np.ndarray
            Predicted ratings.
        """
        ues = self.user_embeddings(users)
        uis = self.item_embeddings(items)

        preds = self.user_biases(users)
        preds += self.item_biases(items)
        preds += (self.dropout(ues) * self.dropout(uis)).sum(
            dim=1, keepdim=True)

        return preds.squeeze()

    def __call__(self, *args):
        return self.forward(*args)

    def predict(self, users, items):
        return self.forward(users, items)

    def _create_embedding(self, n_items, n_factors, pre_weights,
                          pre_biases, freeze, init, sparse, **kwargs):

        bias = nn.Embedding(n_items, 1, sparse=sparse)
        embedding = nn.Embedding(n_items, n_factors, sparse=sparse)
        init(bias.weight.data, **kwargs)
        init(embedding.weight.data, **kwargs)

        if pre_weights is not None:
            embedding.load_state_dict({'weight': pre_weights})

        if pre_biases is not None:
            bias.load_state_dict({'weight': pre_biases})

        if freeze:
            embedding.weight.requires_grad = False
            bias.weight.requires_grad = False

        return embedding, bias


class DenseFFNN(nn.Module):
    """
    Defines the neural network for tabular data
    """

    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories, size) for
             categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 300)
        self.lin2 = nn.Linear(300, 100)
        self.lin3 = nn.Linear(100, 1)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(300)
        self.bn3 = nn.BatchNorm1d(100)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)


    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)

        return x

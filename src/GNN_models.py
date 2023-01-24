import torch
from torch.nn.parameter import Parameter
from torch_geometric_signed_directed.nn import DIMPA
import numpy as np


class DIGRAC_Unroll_Sync(torch.nn.Module):
    r"""The synchronization model with unrolling, adapted from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance" <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.
    Args:
        * **num_features** (int): Number of features.
        * **dropout** (float): Dropout probability.
        * **hop** (int): Number of hops to consider.
        * **embedding_dim** (int) - Embedding dimension.
        * **spectral_step_num** (int, optional) - The number of spectral objactive calculation layers, default 20.
        * **alpha** (float, optional) - (Initial) learning rate for the spectral steps, default 0.01.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
    """

    def __init__(self, num_features: int, dropout: float, hop: int, 
                embedding_dim: int, spectral_step_num: int=20, alpha: float=0.01, 
                trainable_alpha: bool=False):
        super(DIGRAC_Unroll_Sync, self).__init__()
        hidden = int(embedding_dim/2)
        nh1 = hidden
        nh2 = hidden
        self._w_s0 = Parameter(torch.FloatTensor(num_features, nh1))
        self._w_s1 = Parameter(torch.FloatTensor(nh1, nh2))
        self._w_t0 = Parameter(torch.FloatTensor(num_features, nh1))
        self._w_t1 = Parameter(torch.FloatTensor(nh1, nh2))

        self._dimpa = DIMPA(hop)
        self._relu = torch.nn.ReLU()
        self._score_linear = torch.nn.Linear(embedding_dim, 1)
        self._spectral_step_num = spectral_step_num
        self.spectral_layers = torch.nn.ModuleList()
        for _ in range(spectral_step_num):
            self.spectral_layers.append(Projected_Gradient_Step(alpha, trainable_alpha))
        self.dropout = torch.nn.Dropout(p=dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_s0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_s1, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t1, gain=1.414)
    def forward(self, edge_index: torch.FloatTensor, edge_weight: torch.FloatTensor,
                features: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **edge_index** (PyTorch FloatTensor) - Edge indices.
            * **edge_weight** (PyTorch FloatTensor) - Edge weights.
            * **features** (PyTorch FloatTensor) - Input node features, with shape (num_nodes, num_features).
        Return types:
            * **score** (torch.FloatTensor) - Estimated angles.
        """
        # MLP
        x_s = torch.mm(features, self._w_s0)
        x_s = self._relu(x_s)
        x_s = self.dropout(x_s)
        x_s = torch.mm(x_s, self._w_s1)

        x_t = torch.mm(features, self._w_t0)
        x_t = self._relu(x_t)
        x_t = self.dropout(x_t)
        x_t = torch.mm(x_t, self._w_t1)

        z = self._dimpa(x_s, x_t, edge_index, edge_weight)
        score = 2 * np.pi * torch.sigmoid(self._score_linear(z))
        
        # spectral steps to estimate angles
        if self._spectral_step_num > 0:
            A_torch = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=(score.shape[0], score.shape[0])).to_dense()
            # make A skew-symmetric
            A_skewed = A_torch - A_torch.T
            # Map to angular embedding (only the nonzero entries):
            H = torch.exp(1j * A_skewed).multiply(A_skewed != 0)
            for gamma in range(self._spectral_step_num):
                score = self.spectral_layers[gamma](H, score)

        return score
    

class DIGRAC_Unroll_kSync(torch.nn.Module):
    r"""The k-synchronization model with unrolling, adapted from the
    `DIGRAC: Digraph Clustering Based on Flow Imbalance" <https://arxiv.org/pdf/2106.05194.pdf>`_ paper.
    Args:
        * **num_features** (int): Number of features.
        * **dropout** (float): Dropout probability.
        * **hop** (int): Number of hops to consider.
        * **embedding_dim** (int) - Embedding dimension.
        * **k** (int) - Value k for k-synchronization.
        * **spectral_step_num** (int, optional) - The number of spectral objactive calculation layers, default 20.
        * **alpha** (float, optional) - (Initial) learning rate for the spectral steps, default 0.01.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
    """

    def __init__(self, num_features: int, dropout: float, hop: int, 
                embedding_dim: int, k: int, spectral_step_num: int=20, alpha: float=0.01, 
                trainable_alpha: bool=False):
        super(DIGRAC_Unroll_kSync, self).__init__()
        hidden = int(embedding_dim/2) * k
        nh1 = hidden
        nh2 = hidden
        self._emb_dim = embedding_dim
        self._w_s0 = Parameter(torch.FloatTensor(num_features, nh1))
        self._w_s1 = Parameter(torch.FloatTensor(nh1, nh2))
        self._w_t0 = Parameter(torch.FloatTensor(num_features, nh1))
        self._w_t1 = Parameter(torch.FloatTensor(nh1, nh2))

        self.spectral_layers = torch.nn.ModuleList()
        for _ in range(spectral_step_num):
            self.spectral_layers.append(Projected_Gradient_Step(alpha, trainable_alpha))
        self._dimpa = DIMPA(hop)
        self._k = k
        self._relu = torch.nn.ReLU()
        self._score_linear = torch.nn.ModuleList([torch.nn.Linear(embedding_dim, 1) for _ in range(k)])
        self._spectral_step_num = spectral_step_num
        self._trainable_alpha = trainable_alpha
        if self._trainable_alpha:
            self.alpha = Parameter(torch.FloatTensor(1).fill_(alpha))
        else:
            self.alpha = alpha
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_s0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_s1, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t0, gain=1.414)
        torch.nn.init.xavier_uniform_(self._w_t1, gain=1.414)

    def forward(self, edge_index: torch.FloatTensor, edge_weight: torch.FloatTensor,
                features: torch.FloatTensor, separate_graphs: bool=True ) -> torch.FloatTensor:
        """
        Making a forward pass.
        Arg types:
            * **edge_index** (PyTorch FloatTensor) - Edge indices.
            * **edge_weight** (PyTorch FloatTensor) - Edge weights.
            * **features** (PyTorch FloatTensor) - Input node features, with shape (num_nodes, num_features).
            * **separate_graphs** (bool, optional) - Whether to separate edges in the input graphs into k parts for spectal steps. Default False.
        Return types:
            * **score** (torch.FloatTensor) - Estimated angles.
        """
        # MLP
        x_s = torch.mm(features, self._w_s0)
        x_s = self._relu(x_s)
        x_s = self.dropout(x_s)
        x_s = torch.mm(x_s, self._w_s1)

        x_t = torch.mm(features, self._w_t0)
        x_t = self._relu(x_t)
        x_t = self.dropout(x_t)
        x_t = torch.mm(x_t, self._w_t1)

        z = self._dimpa(x_s, x_t, edge_index, edge_weight)
        # to obtain initial scores (estimated angles)
        score = 2 * np.pi * torch.sigmoid(self._score_linear[0](z[:, :self._emb_dim])) # to make nonnegative, range from 0 to 2pi
        for l in range(1, self._k):
            score2 = 2 * np.pi * torch.sigmoid(self._score_linear[l](z[:, l*self._emb_dim:(l+1)*self._emb_dim])) # to make nonnegative, range from 0 to 2pi
            score = torch.cat((score, score2), dim=1)
        
        if self._spectral_step_num > 0 and separate_graphs:
            score_l = score[:, 0:1]
            T = (score_l - score_l.T) % (2*np.pi)
            diff_v1 = (edge_weight-T[edge_index[0], edge_index[1]]) % (2*np.pi)
            diff_v2 = (T[edge_index[0], edge_index[1]]-edge_weight) % (2*np.pi)
            diff = torch.minimum(diff_v1, diff_v2)
            diff_all = diff.clone()[None, :]
            for l in range(1, score.shape[1]):
                score_l = score[:, l:l+1]
                T = (score_l - score_l.T) % (2*np.pi)
                diff_v1 = (edge_weight-T[edge_index[0], edge_index[1]]) % (2*np.pi)
                diff_v2 = (T[edge_index[0], edge_index[1]]-edge_weight) % (2*np.pi)
                diff_l = torch.minimum(diff_v1, diff_v2)
                diff_all = torch.cat((diff_all, diff_l[None, :]), dim=0)
            ind_labels = torch.argmin(diff_all, dim=0)
        
        
        # spectral steps to estimate angles
        if self._spectral_step_num > 0:
            A_torch = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight, size=(score.shape[0], score.shape[0])).to_dense()
            # make A skew-symmetric
            A_skewed = A_torch - A_torch.T
            # Map to angular embedding (only the nonzero entries):
            H = torch.exp(1j * A_skewed).multiply(A_skewed != 0)
            if separate_graphs:
                for l in range(self._k):
                    if (ind_labels==l).sum() > 0:
                        score_l = score[:, l:l+1]
                        H_mask = torch.sparse_coo_tensor(indices=A_torch.nonzero()[ind_labels==l].T, values=torch.ones(len(A_torch.nonzero()[ind_labels==l])).to(A_torch), size=A_torch.shape).to_dense()
                        H_mask = H_mask + H_mask.T
                        H_torch = H.multiply(H_mask > 0)
                        for gamma in range(self._spectral_step_num):
                            score[:, l:l+1] = self.spectral_layers[gamma](H_torch, score[:, l:l+1])
            else:
                for l in range(self._k):
                    for gamma in range(self._spectral_step_num):
                        score[:, l:l+1] = self.spectral_layers[gamma](H, score[:, l:l+1])


        return score

class Projected_Gradient_Step(torch.nn.Module):
    r"""One projected power step for a single graph.

    Args:
        * **alpha** (float, optional) - (Initial) learning rate, default 1.0.
        * **trainable_alpha** (bool, optional) - Whether alpha is trainable, default False.
    """

    def __init__(self, alpha: float=1.0, trainable_alpha: bool=False):
        super(Projected_Gradient_Step, self).__init__()
        self._trainable_alpha = trainable_alpha
        if self._trainable_alpha:
            self.alpha = Parameter(torch.FloatTensor(1).fill_(alpha))
        else:
            self.alpha = alpha
        self._relu = torch.nn.ReLU()

    def forward(self, H: torch.Tensor,
    y: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **H** (PyTorch FloatTensor) - The Hermitian matrix.
            * **y** (PyTorch FloatTensor) - Initial estimated angle vector.

        Return types:
            * **y** (PyTorch FloatTensor) - Updated estimated angle vector.
        """
        if self._trainable_alpha:
            self.alpha = Parameter(self._relu(self.alpha))
        y_complex = torch.exp(y * 1j) # map to complex
        y = torch.angle(self.alpha * y_complex + H @ y_complex) % (2*np.pi) # just like generalized power method

        return y
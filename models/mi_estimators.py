"""
Adapted from https://github.com/Linear95/CLUB/blob/master/mi_estimators.py
"""

import numpy as np
import math
from numbers import Number

import torch
import torch.nn as nn


class CLUBVec2Seq(nn.Module):
    """ The CLUB estimator for vector-to-sequence pairs.
    """

    def __init__(
        self,
        seq_dim: int,
        vec_dim: int,
        hidden_size: int,
        is_sampled_version: bool = False,
    ):
        super().__init__()
        self.is_sampled_version = is_sampled_version

        self.seq_prenet = nn.Sequential(
            nn.Conv1d(seq_dim, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
        )
        # mu net
        self.p_mu = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vec_dim)
        )
        # variance net
        self.p_logvar = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vec_dim),
            nn.Tanh()
        )

    def temporal_avg_pool(self, x, mask=None):
        """
        Args:
            x (tensor): shape [B, T, D]
            mask (bool tensor): padding parts with ones
        """
        if mask is None:
            out = torch.mean(x, dim=1)
        else:
            len_ = (~mask).sum(dim=1).unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(-1), 0)
            x = x.sum(dim=1)
            out = torch.div(x, len_)
        return out

    def get_mu_logvar(self, seq, mask):
        # [B, T, D]
        h = self.seq_prenet(seq.transpose(1, 2)).transpose(1, 2)
        # [B, D]
        h = self.temporal_avg_pool(h, mask)
        mu = self.p_mu(h)
        logvar = self.p_logvar(h)
        return mu, logvar

    def loglikeli(self, seq, vec, mask=None):
        """ Compute un-normalized log-likelihood
        Args:
            seq (tensor): sequence feature, shape [B, T, D].
            vec (tensor): vector feature, shape [B, D].
            mask (tensor): padding parts with ones, [B, T].
        """
        # mu/logvar: (bs, vec_dim)
        mu, logvar = self.get_mu_logvar(seq, mask)
        return (-(mu - vec)**2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def forward(self, seq, vec, mask=None):
        """ Estimate mutual information CLUB upper bound.
        Args:
            seq (tensor): sequence feature, shape [B, T, D].
            vec (tensor): vector feature, shape [B, D].
            mask (tensor): padding parts with ones, [B, T].
        """

        mu, logvar = self.get_mu_logvar(seq, mask)

        if self.is_sampled_version:
            sample_size = seq.shape[0]
            # random_index = torch.randint(sample_size, (sample_size,)).long()
            random_index = torch.randperm(sample_size).long()

            positive = - (mu - vec)**2 / logvar.exp()
            negative = - (mu - vec[random_index])**2 / logvar.exp()
            upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

            mi_upper = upper_bound / 2.
        else:
            # log of conditional probability of positive sample pairs, [B, D]
            positive = - (mu - vec)**2 / 2./logvar.exp()
            # [B, 1, D]
            prediction_1 = mu.unsqueeze(1)
            # [1, B, D]
            y_samples_1 = vec.unsqueeze(0)

            # log of conditional probability of negative sample pairs, [B, D]
            negative = - ((y_samples_1 - prediction_1) **
                          2).mean(dim=1)/2./logvar.exp()

            mi_upper = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # print(mi_upper)

        return mi_upper

    def learning_loss(self, seq, vec, mask=None):
        return - self.loglikeli(seq, vec, mask)


class CLUBForCategorical(nn.Module):  # Update 04/27/2022
    """
    This class provide a CLUB estimator to calculate MI upper bound between 
    vector-like embeddings and categorical labels.
    Estimate I(X,Y), where X is continuous vector and Y is discrete label.
    """

    def __init__(self, input_dim, label_num, hidden_size=None):
        '''
        input_dim : the dimension of input embeddings
        label_num : the number of categorical labels 
        '''
        super().__init__()

        if hidden_size is None:
            self.variational_net = nn.Linear(input_dim, label_num)
        else:
            self.variational_net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, label_num)
            )

    def forward(self, inputs, labels):
        '''
        inputs : shape [batch_size, input_dim], a batch of embeddings
        labels : shape [batch_size], a batch of label index
        '''
        logits = self.variational_net(inputs)  # [sample_size, label_num]

        # log of conditional probability of positive sample pairs
        # positive = - nn.functional.cross_entropy(logits, labels, reduction='none')
        sample_size, label_num = logits.shape

        # shape [sample_size, sample_size, label_num]
        logits_extend = logits.unsqueeze(1).repeat(1, sample_size, 1)
        # shape [sample_size, sample_size]
        labels_extend = labels.unsqueeze(0).repeat(sample_size, 1)

        # log of conditional probability of negative sample pairs
        log_mat = - nn.functional.cross_entropy(
            logits_extend.reshape(-1, label_num),
            labels_extend.reshape(-1, ),
            reduction='none'
        )

        log_mat = log_mat.reshape(sample_size, sample_size)
        positive = torch.diag(log_mat).mean()
        negative = log_mat.mean()
        return positive - negative

    def loglikeli(self, inputs, labels):
        logits = self.variational_net(inputs)
        return - nn.functional.cross_entropy(logits, labels)

    def learning_loss(self, inputs, labels):
        return - self.loglikeli(inputs, labels)


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        This class provides the CLUB estimation to I(X,Y)
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''

    def __init__(self, x_dim, y_dim, hidden_size, is_sampled_version=False):
        super(CLUB, self).__init__()
        self.is_sampled_version = is_sampled_version
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        if self.is_sampled_version:
            sample_size = x_samples.shape[0]
            # random_index = torch.randint(sample_size, (sample_size,)).long()
            random_index = torch.randperm(sample_size).long()

            positive = - (mu - y_samples)**2 / logvar.exp()
            negative = - (mu - y_samples[random_index])**2 / logvar.exp()
            upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()

            mi_upper = upper_bound / 2.
        else:

            # log of conditional probability of positive sample pairs
            positive = - (mu - y_samples)**2 / 2./logvar.exp()

            prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
            y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

            # log of conditional probability of negative sample pairs
            negative = - ((y_samples_1 - prediction_1) **
                          2).mean(dim=1)/2./logvar.exp()

            mi_upper = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        return mi_upper

    def loglikeli(self, x_samples, y_samples):  # unnormalized loglikelihood
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class MINE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(MINE, self).__init__()
        self.T_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    # samples have shape [sample_size, dim]
    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        y_shuffle = y_samples[random_index]

        T0 = self.T_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.T_func(torch.cat([x_samples, y_shuffle], dim=-1))

        lower_bound = T0.mean() - torch.log(T1.exp().mean())

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1))

    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - \
            1.  # shape [sample_size, sample_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size):
        super(InfoNCE, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, 1),
                                    nn.Softplus())

    # samples have shape [sample_size, dim]
    def forward(self, x_samples, y_samples):
        # shuffle and concatenate
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        # [sample_size, sample_size, 1]
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))

        lower_bound = T0.mean() - (T1.logsumexp(dim=1).mean() - np.log(sample_size))
        return lower_bound

    def learning_loss(self, x_samples, y_samples):
        return -self.forward(x_samples, y_samples)


def log_sum_exp(value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation
    value.exp().sum(dim, keepdim).log()
    """
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(torch.exp(value0),
                                       dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        if isinstance(sum_exp, Number):
            return m + math.log(sum_exp)
        else:
            return m + torch.log(sum_exp)


class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples)**2 / 2./logvar.exp() -
                    logvar/2.).sum(dim=-1)  # [nsample]

        mu_1 = mu.unsqueeze(1)          # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)            # [1,nsample,dim]
        all_probs = (- (y_samples_1 - mu_1)**2/2./logvar_1.exp() -
                     logvar_1/2.).sum(dim=-1)  # [nsample, nsample]

        diag_mask = torch.ones(
            [batch_size]).diag().unsqueeze(-1).cuda() * (-20.)
        negative = log_sum_exp(all_probs + diag_mask, dim=0) - \
            np.log(batch_size-1.)  # [nsample]

        return (positive - negative).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class VarUB(nn.Module):  # variational upper bound
    def __init__(self, x_dim, y_dim, hidden_size):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size//2, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size//2),
                                      nn.ReLU(),
                                      nn.Linear(hidden_size//2, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def forward(self, x_samples, y_samples):  # [nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1./2.*(mu**2 + logvar.exp() - 1. - logvar).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 / logvar.exp()-logvar).sum(dim=1).mean(dim=0)

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)


class SymmetricCLUB(nn.Module):
    """
    S-CLUB++ (Symmetric Conditional Log-ratio Upper Bound).

    Unlike the original CLUB, we learn two conditional distributions here:
        q_phi(y|x) and q_psi(x|y).
    This allows us to simultaneously compute:
        U1: the upper bound for x->y,
        U2: the upper bound for y->x,
    in the forward() method.
    We then combine them using a weight alpha to get a symmetric upper bound
    on mutual information.

    Note:
    This is a demonstration implementation. In practical scenarios, you may
    need to consider modeling high-dimensional x and y, random negative sampling,
    and other optimization strategies.
    """

    def __init__(self, x_dim, y_dim, hidden_size, alpha=0.5, is_sampled_version=True):
        """
        Args:
            x_dim (int): Dimensionality of X.
            y_dim (int): Dimensionality of Y.
            hidden_size (int): The size of the hidden layers.
            alpha (float): Weighting factor (0~1) to combine U1 and U2.
                           Defaults to 0.5 for fully symmetric weighting.
            is_sampled_version (bool): Whether to use the version that randomly
                                       permutes indices for negative samples.
        """
        super().__init__()
        self.alpha = alpha
        self.is_sampled_version = is_sampled_version

        # ----- (1) q_phi(y|x) -----
        # p_mu_xy outputs the mean of q_phi(y|x).
        self.p_mu_xy = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
        )
        # p_logvar_xy outputs the log-variance of q_phi(y|x).
        self.p_logvar_xy = nn.Sequential(
            nn.Linear(x_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, y_dim),
            nn.Tanh()  # Similar to the original implementation; range (-1, 1).
        )

        # ----- (2) q_psi(x|y) -----
        # p_mu_yx outputs the mean of q_psi(x|y).
        self.p_mu_yx = nn.Sequential(
            nn.Linear(y_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, x_dim),
        )
        # p_logvar_yx outputs the log-variance of q_psi(x|y).
        self.p_logvar_yx = nn.Sequential(
            nn.Linear(y_dim, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, x_dim),
            nn.Tanh()
        )

    def get_mu_logvar_xy(self, x):
        """Compute the mean and log-variance of y|x."""
        mu = self.p_mu_xy(x)
        logvar = self.p_logvar_xy(x)
        return mu, logvar

    def get_mu_logvar_yx(self, y):
        """Compute the mean and log-variance of x|y."""
        mu = self.p_mu_yx(y)
        logvar = self.p_logvar_yx(y)
        return mu, logvar

    def forward(self, x_samples, y_samples):
        """
        Compute the S-CLUB++ upper bound, which contains two parts:
            U1: (x->y) CLUB
            U2: (y->x) CLUB
        The final upper bound is a weighted combination of both.

        Args:
            x_samples (Tensor): Shape [B, x_dim].
            y_samples (Tensor): Shape [B, y_dim].
        Returns:
            mi_upper (Tensor): A scalar, U_sclub++ = alpha * U1 + (1-alpha) * U2.
        """
        # ------------ (A) Compute U1: x->y CLUB ------------
        mu_xy, logvar_xy = self.get_mu_logvar_xy(x_samples)

        if self.is_sampled_version:
            # Same approach as the original CLUB, with random index permutation
            B = x_samples.size(0)
            random_index = torch.randperm(B).to(x_samples.device)

            positive_xy = - (mu_xy - y_samples) ** 2 / \
                logvar_xy.exp()  # [B, y_dim]
            negative_xy = - \
                (mu_xy - y_samples[random_index]) ** 2 / logvar_xy.exp()
            upper_bound_xy = (positive_xy.sum(dim=-1) -
                              negative_xy.sum(dim=-1)).mean()

            # Divided by 2 as per the original CLUB convention
            U1 = upper_bound_xy / 2.0
            """
            if self.training:  
                print(f"positive_xy.mean = {positive_xy.mean().item():.6f}, "
                    f"negative_xy.mean = {negative_xy.mean().item():.6f}, "
                    f"U1 = {U1.item():.6f}")
            """

        else:
            # Full-pair (non-sampled) computation
            # Log of cond prob for positive pairs
            positive_xy = - (mu_xy - y_samples)**2 / 2. / \
                logvar_xy.exp()  # [B, y_dim]

            # Two-way combination of mu_xy and y_samples
            mu_xy_i = mu_xy.unsqueeze(1)      # [B, 1, y_dim]
            y_j = y_samples.unsqueeze(0)  # [1, B, y_dim]

            # Log cond prob for negative pairs (averaged)
            negative_xy = - ((mu_xy_i - y_j)**2).mean(dim=1) / \
                2. / logvar_xy.exp()

            U1 = (positive_xy.sum(dim=-1) - negative_xy.sum(dim=-1)).mean()
            """
            if self.training:  
                print(f"positive_xy.mean = {positive_xy.mean().item():.6f}, "
                    f"negative_xy.mean = {negative_xy.mean().item():.6f}, "
                    f"U1 = {U1.item():.6f}")
            
        if self.training:
            print("================= DEBUG INFO =================")
            print(f"mu_xy[:5] = \n{mu_xy[:5]}")  
            print(f"y_samples[:5] = \n{y_samples[:5]}")  
            print(f"positive_xy[:5] = \n{positive_xy[:5]}")
            print(f"negative_xy[:5] = \n{negative_xy[:5]}")
        """

        # ------------ (B) Compute U2: y->x CLUB ------------
        mu_yx, logvar_yx = self.get_mu_logvar_yx(y_samples)

        if self.is_sampled_version:
            B = x_samples.size(0)
            random_index = torch.randperm(B).to(x_samples.device)

            positive_yx = - (mu_yx - x_samples) ** 2 / logvar_yx.exp()
            negative_yx = - \
                (mu_yx - x_samples[random_index]) ** 2 / logvar_yx.exp()
            upper_bound_yx = (positive_yx.sum(dim=-1) -
                              negative_yx.sum(dim=-1)).mean()

            U2 = upper_bound_yx / 2.0

        else:
            positive_yx = - (mu_yx - x_samples)**2 / 2. / logvar_yx.exp()

            mu_yx_i = mu_yx.unsqueeze(1)
            x_j = x_samples.unsqueeze(0)

            negative_yx = - ((mu_yx_i - x_j)**2).mean(dim=1) / \
                2. / logvar_yx.exp()
            U2 = (positive_yx.sum(dim=-1) - negative_yx.sum(dim=-1)).mean()

        # ------------ (C) Combine symmetrically ------------
        mi_upper = self.alpha * U1 + (1.0 - self.alpha) * U2
        return mi_upper

    def loglikeli_xy(self, x_samples, y_samples):
        """
        Similar to the original CLUB's log-likelihood, but only for y|x.
        """
        mu_xy, logvar_xy = self.get_mu_logvar_xy(x_samples)
        # -((mu - y)^2 / var + logvar) from the original CLUB:
        return (-(mu_xy - y_samples)**2 / logvar_xy.exp() - logvar_xy).sum(dim=1).mean()

    def loglikeli_yx(self, x_samples, y_samples):
        mu_yx, logvar_yx = self.get_mu_logvar_yx(y_samples)
        return (-(mu_yx - x_samples)**2 / logvar_yx.exp() - logvar_yx).sum(dim=1).mean()

    def learning_loss_symmetric(self, x_samples, y_samples, alpha=0.5):
        ll_xy = self.loglikeli_xy(x_samples, y_samples)  # y|x
        ll_yx = self.loglikeli_yx(x_samples, y_samples)  # x|y
        return -(alpha * ll_xy + (1 - alpha) * ll_yx)

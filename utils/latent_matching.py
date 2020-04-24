"""
Functions for matching latents to sensitive factors
"""

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils import distributions
from utils.math import (log_density_gaussian, log_importance_weight_matrix, 
    matrix_log_density_gaussian, gaussian_entropy)


def GMM_entropy(dist, var, d, bound='upper'):
    # computes bounds for the entropy of a homoscedastic Gaussian mixture model [Kolchinsky, 2017]
    # dist: a matrix of pairwise distances
    # log_var: the log-variance of the mixture components
    # d: number of dimensions
    # n: number of mixture components
    n = tf.cast(tf.shape(dist)[0], tf.float32)
    # var = tf.exp(log_var) + 1e-10

    if bound is 'upper':
        dist_norm = - dist / (2.0 * var)  # uses the KL distance
    elif bound is 'lower':
        dist_norm = - dist / (8.0 * var)  # uses the Bhattacharyya distance
    else:
        print('Error: invalid bound argument')
        return 0

    const = 0.5 * d * tf.log(2.0 * np.pi * np.exp(1.0) * var) + tf.log(n)
    h = const - tf.reduce_mean(tf.reduce_logsumexp(dist_norm, 1))
    return h


def pairwise_distance(x):
    # returns a matrix where each element is the squared distance between each pair of rows in x
    orig_dtype = x.dtype
    
    # these calculations are numerically sensitive, so let's convert to float64
    x = tf.cast(x, tf.float64)
    
    xx = tf.reduce_sum(tf.square(x), 1, keepdims=True)
    dist = xx - 2.0 * tf.matmul(x, tf.transpose(x)) + tf.transpose(xx)
    
    dist = tf.cast(dist, orig_dtype)
    
    dist = tf.nn.relu(dist)  # turn negative numbers into 0 (we only get negatives due to numerical errors)

    return dist

def fast_scaled_gramian(x, logvar):
    """
    Calculates scaled distance (x-x')Σ'^{-1}(x-x')
    for every pair (x,x') in batch. Note asymmetry between x,x'.

    Parameters
    ----------
    x:      torch.Tensor
            Matrix containing observations as rows, [B,D]
    logvar: torch.Tensor
            Matrix containing log-variance as rows, [B,D]
    """
    inv_var_diag_sqrt = torch.exp(-0.5* logvar)
    inv_var_diag = inv_var_diag_sqrt * inv_var_diag_sqrt
    x_j_sqrt_scaled = inv_var_diag_sqrt * x
    x_j_scaled = inv_var_diag * x
    x_scale_j_x = x**2 @ inv_var_diag.T 
    x_jx_j = (x_j_sqrt_scaled * x_j_sqrt_scaled).sum(dim=1, keepdim=True)

    scaled_distance = x_scale_j_x - 2.0 * x @ x_j_scaled.T + x_jx_j.T

    return torch.clamp(scaled_distance, min=0.)
    

def slow_scaled_gramian(x, logvar):
    """
    More interpretable function to calculate scaled distance 
    (x-x')Σ'^{-1}(x-x')
    """
    inv_var_diag = torch.exp(-logvar)
    y_slow = torch.empty(len(x),len(x))
    for i in range(len(x)):
        for j in range(len(x)):
            y_slow[i][j] = torch.dot(x[i], inv_var_diag[j] * x[i]) - 2.0 * torch.dot(x[i], inv_var_diag[j] * x[j]) + torch.dot(x[j], inv_var_diag[j] * x[j])
    return y_slow

def mahalanobis_distance(x, logvar):
    """
    Returns a matrix where each element is the Mahalanobis distance between each pair of rows in x,
    assuming diagonal covariance, with covariance of each dimension shared among all examples. 
    D_{ij} = (x_i - x_j)^T Σ^{-1} (x_i - x_j)

    Parameters
    ----------
    x:      torch.Tensor [batch_size, D]
    logvar:  torch.Tensor [batch_size, D] - Diagonal of log-variance matrix
    """
    x = x.double()
    logvar = logvar.double()

    sqrt_Sigma_inv_diag = torch.exp(-0.5 * logvar)
    x = x * sqrt_Sigma_inv_diag
    xx = (x*x).sum(1, keepdim=True)
    gramian = x @ x.T

    m_dist = xx - 2.0 * gramian + xx.T

    return torch.clamp(m_dist, min=0.)

def slow_pairwise_kl(x, logvar):
    B, D = x.size()

    inv_var_diag = torch.exp(-logvar)
    y_slow = torch.empty(len(x),len(x))

    for i in range(len(x)):
        for j in range(len(x)):
            scaled_G_ij = torch.dot(x[i], inv_var_diag[j] * x[i]) - 2.0 * torch.dot(x[i], inv_var_diag[j] * x[j]) + torch.dot(x[j], inv_var_diag[j] * x[j])
            y_slow[i][j] = 0.5 * ( torch.sum(logvar[j]) - torch.sum(logvar[i]) + scaled_G_ij + torch.sum(torch.exp(logvar[i] - logvar[j])) - D )

    return y_slow


def pairwise_kl(x, logvar):
    """
    Calculates KL between two diagonal-covariance Gaussians for all combination of 
    batch pairs of `logvar` and `mu`. i.e. return tensor of shape `(batch_size, batch_size)`

    Parameters
    ----------
    x:      torch.Tensor [batch_size, D]
    logvar:  torch.Tensor [batch_size, D] - Diagonal of log-variance matrix
    Returns
    ----------
    pairwise_KLD torch.Tensor [batch_size, batch_size]
    """
    x = x.double()
    logvar = logvar.double()

    B, D = x.size()

    tr_logvar = torch.sum(logvar, dim=1)  # [B]
    tr_logvar_ij = -tr_logvar.view(B,1) + tr_logvar.view(1,B)

    # var_diag = torch.exp(logvar)  # [B,D]
    # tr_ratio_var_ij = torch.div(var_diag.view(B,1,D) - var_diag.view(1,B,D)).sum(dim=2)
    tr_ratio_var_ij = torch.exp(logvar.view(B,1,D) - logvar.view(1,B,D)).sum(dim=2)

    mld_ij = fast_scaled_gramian(x, logvar)

    pairwise_KLD = 0.5 * (tr_logvar_ij + mld_ij + tr_ratio_var_ij - D)

    return torch.clamp(pairwise_KLD, min=0.)

def MI_matching(latent_stats, generative_factors, sensitive_latent_idx, latent_factors, storage=None):
    """
    Supervised loss to match latent factors z with known generative
    factors v. 
    Parameters
    ----------
    latent_stats: torch.Tensor
        Mean, log-variance of latent distribution. Shape (batch_size, latent_dim)
        Distribution is assumed Gaussian.
    latent_factors: torch.Tensor
        Sample drawn from latent distribution Shape (batch_size, latent_dim)
    generative_factors: torch.Tensor
        True known generative factors of data. Shape (batch_size, latent_dim)
    sensitive_latent_idx: list
        Indices of sensitive latents. Advised to use {0,1,...,d} for d
        sensitive latents.
    Returns
    -------
    Nonparametric lower bound on mutual information between inputs (learned latents) z and targets 
    (ground truth generative factors) v, as proposed in [1].

    [1]: Estimating Mixture Entropy with Pairwise Distances, Artemy Kolchinsky, Brendan D. Tracey,
         arxiv.org/abs/1706.02419
    """
    mu, logvar = latent_stats
    B,D = mu.size()

    latent_factors = latent_factors.float()
    generative_factors = generative_factors.float()
    batch_size, latent_dim = latent_factors.shape

    pairwise_kl_distance = pairwise_kl(mu, logvar)

    # Upper bound on 
    I_XZ = -torch.mean(torch.logsumexp(-pairwise_kl_distance, dim=1)) + math.log(B)


    return matching_term

def cross_entropy_matching(latent_stats, generative_factors, sensitive_latent_idx, latent_factors=None, storage=None):
    """
    Supervised loss to match latent factors z with known generative
    factors v. 
    Parameters
    ----------
    latent_factors: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        or mean of latent distribution. Shape (batch_size, latent_dim)
    generative_factors: torch.Tensor
        True known generative factors of data. Shape (batch_size, latent_dim)
    sensitive_latent_idx: list
        Indices of sensitive latents. Advised to use {0,1,...,d} for dfr
        sensitive latents.
    Returns
    -------
    Binary cross entropy loss between inputs (learned latents) z and targets 
    (ground truth generative factors) v normalized to [0,1].

    \sum_i^d v_i \log (\sigma z_i) + (1-v_i) \log (1 - \sigma z_i) 
    """

    mu = latent_stats[0].float()
    latent_factors = mu
    generative_factors = generative_factors.float()

    rep = latent_factors
    sigmoid_rep = torch.sigmoid(rep)

    batch_size, latent_dim = latent_factors.shape

    # For some reason this is returning divergent values and nondeterministic
    # supervised_term = torch.stack([F.binary_cross_entropy_with_logits(input=latent_factors[:,idx], 
    #     target=generative_factors[:,idx], reduction='sum') for idx in sensitive_latent_idx]).sum()

    supervised_term = torch.stack([F.binary_cross_entropy(input=sigmoid_rep[:,idx], 
        target=generative_factors[:,idx], reduction='sum') for idx in sensitive_latent_idx]).sum()

    supervised_term = supervised_term / batch_size

    if storage is not None:
        storage['supervised_term'].append(supervised_term.item())

    return supervised_term


MATCHING_FUNCTIONS = {'cross_entropy': cross_entropy_matching, 'mutual_info': MI_matching}

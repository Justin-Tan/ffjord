""" Authors: @YannDubs 2019
             @sksq96   2019 """

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
import os, time
import logging
from collections import OrderedDict

def traverse_continuous_grid(idx, size=(10,10), axis=1, latent_dim=10, sample_prior=False):
    """
    Returns a (size[0] * size[1], cont_dim) latent sample, corresponding to
    a two dimensional traversal of the continuous latent space.
    Parameters
    ----------
    idx : int or None
        Index of continuous latent dimension to traverse. If None, no
        latent is traversed and all latent dimensions are randomly sampled
        or kept fixed.
    axis : int
        Either 0 for traversal across the rows or 1 for traversal across
        the columns.
    size : tuple of ints
        Shape of grid to generate. E.g. (6, 4).
    """
    num_samples = size[0] * size[1]

    if sample_prior:
        samples = np.random.normal(size=(num_samples, latent_dim))
    else:
        samples = np.zeros(shape=(num_samples, latent_dim))

    if idx is not None:
        # Sweep over linearly spaced coordinates transformed through the
        # inverse CDF (ppf) of a gaussian since the prior of the latent
        # space is gaussian
        cdf_traversal = np.linspace(0.05, 0.95, size[axis])
        cont_traversal = stats.norm.ppf(cdf_traversal)

        for i in range(size[0]):
            for j in range(size[1]):
                if axis == 0:
                    samples[i * size[1] + j, idx] = cont_traversal[i]
                else:
                    samples[i * size[1] + j, idx] = cont_traversal[j]

    return torch.Tensor(samples)


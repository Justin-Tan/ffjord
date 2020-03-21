"""
Module containing different loss implementations

Based on Torch implementation by @yanndubs + @rtiqchen
"""

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from models import network
from utils import distributions
from utils.math import (log_density_gaussian, log_importance_weight_matrix, 
    matrix_log_density_gaussian)

def get_loss_function(loss_type, args, **kwargs):
    """Return the correct loss function given the argparse arguments."""    
    if loss_type == "VAE":
        return BetaVAE_loss(beta=1.0, **kwargs)
    elif loss_type == "beta_VAE":
        return BetaVAE_loss(beta=args.beta, **kwargs)
    elif loss_type == "annealed_VAE":
        return AnnealedVAE_loss(C_init=args.C_init, C_fin=args.C_fin, gamma=args.gamma, **kwargs)
    elif loss_type == "factor_VAE":
        return FactorVAE_loss(gamma=args.gamma_fvae, disc_kwargs=dict(n_layers=args.n_layers_D, 
            latent_dim=args.latent_dim, n_units=args.n_units_D), 
            optim_kwargs=dict(lr=args.lr_D, betas=(0.5,0.9)), **kwargs)
    elif loss_type == "beta_TCVAE":
        return betaTC_VAE_loss(n_data=args.n_data, alpha=args.alpha_btcvae, beta=args.beta_btcvae, 
            gamma=args.gamma_btcvae, **kwargs)
    elif loss_type == "beta_TCVAE_sensitive":
        return betaTC_sensitive_VAE_loss(n_data=args.n_data, alpha=args.alpha_btcvae, beta=args.beta_btcvae, 
                gamma=args.gamma_btcvae, delta=1., is_mss=True, **kwargs)
    else:
        raise ValueError("Unknown loss : {}. \n Supported losses: {}".format(loss_type,
            "[VAE, beta_VAE, annealed_VAE, factor_VAE. beta_TCVAE, beta_TCVAE_sensitive]"))


class BaseLoss:
    
    def __init__(self, x_dist, prior, supervision=False, distribution="bernoulli", log_interval=100, annealing_steps=1e4,
            supervision_lagrange_m=64, sensitive_latent_idx=None, flow_type='no_flow', **kwargs):
        self.prior = prior  # Prior over z, p(z)
        self.x_dist = x_dist  # Distribution input x, p(x|z), for non-image data
        self.distribution = distribution  # x_dist but for image data - involves certain heuristic modifications
        self.train_steps = 0
        self.log_interval = log_interval
        self.annealing_steps = annealing_steps
        self.supervision = supervision
        self.supervision_lagrange_m = supervision_lagrange_m
        self.flow = flow_type

        try:
            sensitive_latent_idx.sort()
            self.sensitive_latent_idx = sensitive_latent_idx
        except AttributeError:
            pass

        if self.supervision is True:
            assert sensitive_latent_idx is not None, 'Must supply sensitive latent indices!'
            assert self.supervision_lagrange_m > 0, 'Warning: Supervision coefficient is zero.'
            print("Using supervised loss over dimensions", self.sensitive_latent_idx)
        
    def __call__(self, data, reconstruction, latent_stats, storage, training=True, **kwargs):
        
        """
        Data / Recon: Original data and decoder-reconstructed output
            Shape [B,C,H,W]
        latent_stats: Sufficient statistics of latent representation. e.g. (mu, logvar)
            Shape [B,latent_dim]
        """
        
    def _precall(self, training, storage):
        if training:

            self.train_steps += 1
            
        if (self.train_steps % self.log_interval == 1 and self.train_steps > 1) or (not training):
            storage = storage
        else:
            storage = None
        
        return storage
    
    def _record_losses(self, storage, total_loss, recon_loss, kl_loss):
        storage['total_loss'].append(total_loss.item())
        storage['reconstruction_loss'].append(recon_loss.item())
        storage['kl_loss'].append(kl_loss.item())
        storage['ELBO'].append(-recon_loss.item() - kl_loss.item())

class BetaVAE_loss(BaseLoss):
    """
    Compute the Beta-VAE loss as in [1]
    Parameters
    ----------
    beta : float, optional
        Weight of the KL divergence.
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """
    
    def __init__(self, beta=4.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        
    def __call__(self, data, reconstruction, latent_stats, storage, training=True, 
            latent_sample=None, generative_factors=None, flow_output=None, **kwargs):
        storage = self._precall(training, storage)

        recon_loss = _reconstruction_loss(data, reconstruction, distribution=self.distribution,
                x_dist=self.x_dist, x_stats=reconstruction, flow_type=self.flow, flow_output=flow_output,
                latent_stats=latent_stats, latent_sample=latent_sample)
        kl_loss = _kl_divergence_q_prior_normal(*latent_stats)
        loss = recon_loss + self.beta * kl_loss
        
        if self.supervision is True:
            assert generative_factors is not None, 'Must supply generative factors for supervision.'
            supervised_term = _sensitive_generative_matching(latent_stats, generative_factors, self.sensitive_latent_idx, storage=storage)
            loss += self.supervision_lagrange_m * supervised_term

        if storage is not None:
            self._record_losses(storage, loss, recon_loss, kl_loss)
        
        return loss
    
class AnnealedVAE_loss(BaseLoss):
    """
    Compute the Beta-VAE variant loss as in [1].
    Allows both discrete and continuous latent factors,
    after [2]
    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.
    C_fin : float, optional
        Final annealed capacity C.
    gamma : float, optional
        Weight of the KL divergence term.
    latent_dist : dict, torch.Tensor values
        Dict containing keys 'continuous', possibly 'discrete' containing 
        the parameters of the latent distributions as torch.Tensor instances.
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
        [2] Dupont, Emilien. "Learning Disentangled Joint Continuous and 
        Discrete Representations." arXiv preprint arXiv:1804.00104 (2018).
    """
    
    def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
        super().__init__(**kwargs)

        self.gamma = gamma
        self.C_init = C_init
        self.C_fin = C_fin
        
    def __call__(self, data, reconstruction, latent_stats, storage, training=True, 
            latent_sample=None, generative_factors=None, flow_output=None, **kwargs):
        storage = self._precall(training, storage)

        latent_dist = kwargs['latent_dist']
        latent_stats_normal = latent_dist['continuous']

        recon_loss = _reconstruction_loss(data, reconstruction, distribution=self.distribution,
                x_dist=self.x_dist, x_stats=reconstruction, flow_type=self.flow, flow_output=flow_output,
                latent_stats=latent_stats, latent_sample=latent_sample)
        kl_loss_normal = _kl_divergence_q_prior_normal(*latent_stats_normal)
        C = (_linear_annealing(self.C_init, self.C_fin, self.train_steps, self.annealing_steps) 
             if training else self.C_fin)

        kl_loss = kl_loss_normal
        # loss = recon_loss + self.gamma * torch.abs(kl_loss_normal - C) + self.gamma * torch.abs(kl_loss_discrete - C_discrete)
        loss = recon_loss + self.gamma * torch.abs(kl_loss_normal - C)

        if self.supervision is True:
            assert generative_factors is not None, 'Must supply generative factors for supervision.'
            supervised_term = _sensitive_generative_matching(latent_stats, generative_factors, self.sensitive_latent_idx, storage=storage)
            loss += self.supervision_lagrange_m * supervised_term

        if 'discrete' in latent_dist.keys():
            latent_stats_discrete = latent_dist['discrete']
            kl_loss_discrete_dimwise = [_kl_divergence_q_prior_uniform(alpha) for alpha in latent_stats_discrete]
            kl_loss_discrete = torch.sum(torch.cat(kl_loss_discrete_dimwise))
            C_discrete = (_linear_annealing(self.C_init, self.C_fin, self.train_steps, self.annealing_steps) 
                 if training else self.C_fin)

            kl_loss += kl_loss_discrete
            loss += self.gamma * torch.abs(kl_loss_discrete - C_discrete)
        
        if storage is not None:
            self._record_losses(storage, loss, recon_loss, kl_loss_normal)
            storage['kl_loss_total'].append(kl_loss.item())
            if 'discrete' in latent_dist.keys():
                storage['kl_loss_discrete'].append(kl_loss_discrete.item())

        
        return loss


    
class FactorVAE_loss(BaseLoss):
    """
    Compute the Factor-VAE loss by density estimation
    Parameters
    ----------
    device : torch.device
    gamma : float, optional
        Weight of the TC loss term. `gamma` in the paper.
    discriminator : Torch module representing discriminator
    optimizer_d : torch.optim
    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    References
    ----------
        [1] Kim, Hyunjik, and Andriy Mnih. "Disentangling by factorising."
        arXiv preprint arXiv:1802.05983 (2018).
    """
    def __init__(self, device, gamma=10., disc_kwargs={}, optim_kwargs=dict(lr=5e-5, betas=(0.5, 0.9)), **kwargs):
        super().__init__(**kwargs)

        self.gamma = gamma
        self.device = device
        self.discriminator = network.Discriminator(**disc_kwargs).to(self.device)
        self.CE_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.opt_D = optim.Adam(self.discriminator.parameters(), **optim_kwargs)
        
    def __call__(self, data, reconstruction, latent_stats, storage, training=True, **kwargs):
        raise ValueError("Use `call_optimize` to also train the discriminator")

    def call_optimize(self, data, model, storage, optimizer, training=True, generative_factors=None, 
        flow_output=None, **kwargs):
        """
        Return Factor-VAE augmented loss and 
        optimize Factor-VAE Discriminator for 1 iteration.

        """
        storage = self._precall(training, storage)

        # Split batch in half - try to use batch size 128 or above
        batch_size = data.size(0)
        half_batch_size = batch_size // 2
        data_VAE, data_D = data.split(half_batch_size)
        gen_f_VAE, _ = generative_factors.split(half_batch_size)
        
        reconstruction, latent_sample, latent_dist, flow_output = model(data_VAE)
        latent_stats = latent_dist['continuous']

        recon_loss = _reconstruction_loss(data, reconstruction, distribution=self.distribution,
                x_dist=self.x_dist, x_stats=reconstruction, flow_type=self.flow, flow_output=flow_output,
                latent_stats=latent_stats, latent_sample=latent_sample)
        kl_loss = _kl_divergence_q_prior_normal(*latent_stats)
        
        loss = recon_loss + kl_loss
        
        # Output logit D(z) - \sigmoid(D(z)) is probability that the input is 
        # sample q(z) rather than permuted q(\bar{z})
        D_z = self.discriminator(latent_sample)

        # Softmax -> D_z.shape = (B, 2)
        TC_loss = (D_z[:,0] - D_z[:,1]).flatten().mean()
        # Sigmoid -> D_z.shape = (B, 1) -> (inverse logit)
        # TC_loss = D_z.flatten().mean()

        anneal_reg = (_linear_annealing(0, 1, self.train_steps, self.annealing_steps) 
                if model.training else 1)

        vae_loss = recon_loss + kl_loss + anneal_reg * self.gamma * TC_loss 

        if self.supervision is True:
            assert generative_factors is not None, 'Must supply generative factors for supervision.'
            supervised_term = _sensitive_generative_matching(latent_stats, gen_f_VAE, self.sensitive_latent_idx, storage=storage)
            vae_loss += self.supervision_lagrange_m * supervised_term

        if storage is not None:
            #print('q(z|x) logvar', latent_stats[-1].mean().item())
            #print('p(x|z) logvar', reconstruction['logvar'].mean().item())
            self._record_losses(storage, vae_loss, recon_loss, kl_loss)
            storage['TC_loss'].append(TC_loss.item())

        if model.training is False:
            return vae_loss

        # 1 gradient step for VAE model
        optimizer.zero_grad()
        vae_loss.backward(retain_graph=True)
        optimizer.step()

        # Now update discriminator using Bernoulli proper scoring rule
        try:
            latent_stats_D = model.encoder(data_D)
            latent_sample_D = model.reparameterize(latent_stats_D)
        except AttributeError:
            latent_stats_D = model.module.encoder(data_D)
            latent_sample_D = model.module.reparameterize(latent_stats_D)            

        z_perm = _permute_dims(latent_sample_D).detach()
        D_z_perm = self.discriminator(z_perm)
        ones = torch.ones(half_batch_size, dtype=torch.long, device=self.device)
        zeros = torch.zeros_like(ones)
        D_loss = 0.5 * (self.CE_loss(D_z, zeros) + self.CE_loss(D_z_perm, ones))
        # For sigmoid loss
        # D_loss = self.BCE_loss(D_z.flatten(), ones) + self.BCE_loss(D_z_perm.flatten(), zeros)

        # 1 gradient step for discriminator
        self.opt_D.zero_grad()
        D_loss.backward()
        self.opt_D.step()

        if storage is not None:
            storage['D_loss'].append(D_loss.item())

        return vae_loss

class betaTC_VAE_loss(BaseLoss):
    
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]
    Parameters
    ----------
    n_data: int
        Number of instances in the training set
    alpha : float
        Weight of the mutual information term.
    beta : float
        Weight of the total correlation term.
    gamma : float
        Weight of the dimension-wise KL term.
    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.
    sensitive_latent_idx: list
        Indices of sensitive latents. Advised to use {0,1,...,n} for n
        sensitive latents.
    supervision: bool
        Whether to use supervised loss - need to have true generative factors.
        Only works for DSprites dataset currently. 
    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """
    
    def __init__(self, n_data, alpha=1., beta=6., gamma=1., is_mss=True, **kwargs):

        super().__init__(**kwargs)
        self.n_data = n_data
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.is_mss = is_mss 

    def __call__(self, data, reconstruction, latent_stats, storage, latent_sample=None, 
            generative_factors=None, flow_output=None, training=True, **kwargs):
        storage = self._precall(training, storage)
        recon_loss = _reconstruction_loss(data, reconstruction, distribution=self.distribution,
                x_dist=self.x_dist, x_stats=reconstruction, flow_type=self.flow, flow_output=flow_output,
                latent_stats=latent_stats, latent_sample=latent_sample)
        kl_loss, kl_loss_z = _kl_divergence_q_prior_normal(*latent_stats, per_dim=True)

        log_pz, log_qz, log_prod_qzi, log_q_zCx = _get_log_pz_qz_prodzi_qzCx(latent_sample, 
            latent_stats, n_data=self.n_data, is_mss=self.is_mss, prior=self.prior)
        
        # MI b/w latent and data point I(z;x) = KL(q(z,x) || q(x)q(z)) = E_x[KL(q(z|x) || q(z))]
        I_xz = (log_q_zCx - log_qz).mean() 
        # TC loss KL( q(\vec{z}) || \prod_i q(\vec{z}_i))
        tc_loss = (log_qz - log_prod_qzi).mean()
        # Dimwise KL-loss \sum_j KL(q(z_j) || p(z_j)) = KL(q(z) || p(z))
        dw_kl_loss = (log_prod_qzi - log_pz).mean()

        anneal_reg = (_linear_annealing(0, 1, self.train_steps, self.annealing_steps) 
                if training else 1)
        
        # loss = recon_loss + self.alpha * I_xz + self.beta * tc_loss + anneal_reg * self.gamma * dw_kl_loss
        loss = recon_loss + self.alpha * I_xz + self.beta * tc_loss + self.gamma * dw_kl_loss

        if self.supervision is True:
            assert generative_factors is not None, 'Must supply generative factors for supervision.'
            supervised_term = _sensitive_generative_matching(latent_stats, generative_factors, self.sensitive_latent_idx, storage=storage)
            loss += self.supervision_lagrange_m * supervised_term

        if storage is not None:
            #print('q(z|x) logvar', latent_stats[-1].mean().item())
            #print('p(x|z) logvar', reconstruction['logvar'].mean().item())
            self._record_losses(storage, loss, recon_loss, kl_loss)
            storage['mi_loss'].append(I_xz.item())
            storage['tc_loss'].append(tc_loss.item()) 
            storage['dw_kl_loss'].append(dw_kl_loss.item())

            for dim, kl_loss_dim in enumerate(kl_loss_z):
                storage['kl_loss_z{}'.format(dim)].append(kl_loss_dim.item())

        return loss

class betaTC_sensitive_VAE_loss(BaseLoss):
    
    """
    Compute the decomposed KL loss with either minibatch weighted sampling or
    minibatch stratified sampling according to [1]
    Parameters
    ----------
    n_data: int
        Number of instances in the training set
    alpha : float
        Weight of the mutual information term.
    beta : float
        Weight of the total correlation term.
    gamma : float
        Weight of the dimension-wise KL term.
    is_mss : bool
        Whether to use minibatch stratified sampling instead of minibatch
        weighted sampling.
    sensitive_latent_idx: list
        Indices of sensitive latents. Advised to use {0,1,...,n} for n
        sensitive latents.
    supervision: bool
        Whether to use supervised loss - need to have true generative factors.
        Only works for DSprites dataset currently. 
    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.
    References
    ----------
       [1] Chen, Tian Qi, et al. "Isolating sources of disentanglement in variational
       autoencoders." Advances in Neural Information Processing Systems. 2018.
    """
    
    def __init__(self, n_data, alpha=1., beta=6., gamma=1., delta=1., 
            is_mss=True, **kwargs):
        super().__init__(**kwargs)

        self.n_data = n_data
        self.alpha = alpha

        self.beta = beta
        self.beta_ns = 1.
        self.beta_s = self.beta

        self.gamma = gamma
        self.delta = delta
        self.is_mss = is_mss 

       
    def __call__(self, data, reconstruction, latent_stats, storage, latent_sample, 
            generative_factors=None, flow_output=None, training=True, **kwargs):
        
        storage = self._precall(training, storage)
        recon_loss = _reconstruction_loss(data, reconstruction, distribution=self.distribution,
                x_dist=self.x_dist, x_stats=reconstruction, flow_type=self.flow, flow_output=flow_output,
                latent_stats=latent_stats, latent_sample=latent_sample)
        kl_loss, kl_loss_z = _kl_divergence_q_prior_normal(*latent_stats, per_dim=True)

        elbo_terms = _get_log_pz_qz_prodzi_qzCx_isolate_sensitives(latent_sample, latent_stats, 
                n_data=self.n_data, sensitive_latent_idx=self.sensitive_latent_idx, is_mss=self.is_mss)
        log_pz, log_pt, log_pz_minus_t, log_qz, log_q_z_minus_t, log_prod_qzi, log_prod_qti, log_prod_qz_minus_ti, log_q_zCx, mi_components = elbo_terms

        # Supervised loss part - matching latents to generative factors
        # Try using both mean and sample as inputs
        
        log_q_tCx = mi_components['log_q_tCx']
        log_q_z_minus_tCx = mi_components['log_q_z_minus_tCx']
        log_qt = mi_components['log_qt']

        # MI b/w latent and data point I(z;x) = KL(q(z,x) || p(x)q(z)) = E_x[KL(q(z|x) || q(z))]
        I_xz = (log_q_zCx - log_qz).mean()
        I_x_z_minus_t = (log_q_z_minus_tCx - log_q_z_minus_t).mean()
        I_xt = (log_q_tCx - log_qt).mean()
        # TC loss KL( q(\vec{z}) || \prod_i q(\vec{z}_i))
        tc_loss = (log_qz - log_prod_qzi).mean()
        tc_loss_sensitives = (log_qz - log_q_z_minus_t - log_prod_qti).mean()
        tc_loss_nonsensitives = (log_q_z_minus_t - log_prod_qz_minus_ti).mean()
        # Dimwise KL-loss \sum_j KL(q(z_j) || p(z_j)) = KL(q(z) || p(z))
        dw_kl_loss = (log_prod_qzi - log_pz).mean()
        dw_kl_loss_sensitives = (log_prod_qti - log_pt).mean()

        kl_loss_nonsensitive = (log_q_z_minus_t - log_pz_minus_t).mean()

        anneal_reg = (_linear_annealing(0, 1, self.train_steps, self.annealing_steps) 
                if training else 1)
        
        # loss = recon_loss + self.alpha * I_xz + self.beta * tc_loss + anneal_reg * self.gamma * dw_kl_loss
        # loss = recon_loss + self.alpha * I_xz + self.beta * tc_loss + self.gamma * dw_kl_loss
        
        # KL(q(z|x)||p(z)), with reweighted components
        btcvae_kl_term = self.alpha * I_xz + self.beta * tc_loss + self.gamma * dw_kl_loss

        # Use original TC term, weight nonsensitive/sensitive components separately
        # Use original dimwise KL loss as well
        kl_term = self.alpha * I_xz + (self.beta_ns * tc_loss_nonsensitives + self.beta_s * tc_loss_sensitives) + self.gamma * dw_kl_loss
        kl_term_original = self.alpha * I_xz + self.beta * tc_loss_sensitives + self.gamma * (dw_kl_loss_sensitives + kl_loss_nonsensitive)
        kl_term_original = self.alpha * I_xt + self.beta * tc_loss_sensitives + self.gamma * (dw_kl_loss_sensitives + kl_loss_nonsensitive)
        # kl_term_original = self.alpha * (I_xt - I_x_z_minus_t) + self.beta * tc_loss_sensitives + self.gamma * (dw_kl_loss_sensitives + kl_loss_nonsensitive)

        loss = recon_loss  + kl_term_original  # + kl_term_original  # kl_term_original  # kl_term
        
        if self.supervision is True:
            assert generative_factors is not None, 'Must supply generative factors for supervision.'
            supervised_term = _sensitive_generative_matching(latent_stats, generative_factors, self.sensitive_latent_idx, storage=storage)
            loss += self.supervision_lagrange_m * supervised_term

        if storage is not None:
            #print('q(z|x) logvar', latent_stats[-1].mean().item())
            #print('p(x|z) logvar', reconstruction['logvar'].mean().item())
            self._record_losses(storage, loss, recon_loss, kl_loss)
            storage['mi_loss'].append(I_xz.item())
            storage['tc_loss'].append(tc_loss.item()) 
            storage['tc_loss_sensitives'].append(tc_loss_sensitives.item())
            storage['dw_kl_loss'].append(dw_kl_loss.item())
            storage['dw_kl_loss_sensitives'].append(dw_kl_loss_sensitives.item())
            
            for dim, kl_loss_dim in enumerate(kl_loss_z):
                storage['kl_loss_z{}'.format(dim)].append(kl_loss_dim.item())

        return loss

def _reconstruction_loss(data, reconstruction=None, reconstruction_logits=None, distribution="bernoulli",
        x_dist=None, x_stats=[], flow_type='no_flow', flow_output=None, latent_sample=None, latent_stats=None):
    """
    Returns negative log likelihood under the variational posterior:
    $ - E_{q_{\phi}(z|x)}[\log p_{\theta}(x|z) $
    This forms the 'reconstruction loss' part of the ELBO.
    
    distribution: {"bernoulli", "gaussian", "laplace"}
    Distribution of the likelihood over each pixel. We can compute
    the likelihood in closed form for the absove distributions. 

    Parameters (some)
    ----------------- 
    data            :  torch.Tensor - Batch of training instances x
    reconstruction  :  torch.Tensor - Output of the decoder network p(x|z) for image data
    distribution    :  string - String specifying distribution of input x for image data
    x_dist          :  class - Object specifying distribution of input x for non-image data
    x_stats         :  list - List containing distribution parameters, input to x_dist,
                       output of the decoder network p(x|z) for non-image data
    """
    

    image_data = (len(data.shape) > 2)
    
    if image_data is True:
        assert data.shape == reconstruction.shape, 'Reconstruction and data must have identical shapes!'
        batch_size, n_channels, height, width = data.shape
        if distribution == "bernoulli":
            loss = F.binary_cross_entropy(reconstruction, data, reduction='sum')
    #         loss = F.binary_cross_entropy_with_logits(input=reconstruction_logits, 
    #             target=data, reduction='sum')
        elif distribution == "gaussian":
            # loss in [0,255] space but normalized by 255 to not be too big
            loss = F.mse_loss(input=reconstruction * 255, target=data * 255, reduction="sum") / 255
        elif distribution == "laplace":
            loss = F.l1_loss(input=reconstruction, target=data, reduction="sum")
            loss = loss * 3  # empirical value to give similar values than bernoulli => use same hyperparam
            loss = loss * (loss != 0)  # maskimg to avoid nan
        else:
            raise ValueError("Unknown distribution: {}".format(distribution))
        loss = loss / batch_size  # normalize over batch dimension

    else:
        # Assume data is of shape [batch_size, x_dim]
        batch_size, x_dim = data.shape

        if flow_type == 'no_flow':
            # Poor assumption: data can be modelled by diagonal-covariance Gaussian
            assert x_dist is not None, 'Must specify distribution of inputs x'
            # Have to be careful with ordering of distribution parameters when unpacking
            log_pxCz = x_dist.log_density(data, mu=x_stats['mu'], logvar=x_stats['logvar']).view(batch_size, -1).sum(1)
            # print(log_px)
        else:
            assert flow_output is not None, 'Must specify normalizing flow'
            if flow_type == 'real_nvp':
                x_flow_inv, log_det_jacobian_inv = flow_output['x_flow_inv'], flow_output['log_det_jacobian_inv']
                assert log_det_jacobian_inv is not None, 'Must supply determinant of transformation Jacobian!'
                log_pxCz = _flow_log_density(x_flow_inv, x_dist, x_stats, log_det_jacobian_inv)
            elif flow_type == 'cnf':  # use VAE-ODE class 
                x_flow, delta_logp = flow_output['x_flow'], flow_output['log_det_jacobian']
                assert delta_logp is not None, 'Must supply determinant of transformation Jacobian!'
                log_pxCz = _ffjord_log_density(x_flow, delta_logp, x_stats)

        loss = -log_pxCz.mean()

    return loss

def _ffjord_log_density(x_flow, delta_logp, x_stats):

    # Compute log-prob of sample from base distribution
    log_p0_xCz = log_density_gaussian(x_flow, *x_stats).sum(dim=1)

    # Compute log-prob of transformed data
    log_pxCz = log_p0_xCz - delta_logp

    return log_pxCz

def _flow_log_density(x_flow_inv, x_dist, x_stats, log_det_jacobian_inv):

    """
    Let \hat{x} = f_k \circ f_{k-1} \circ ... \circ f_1 (x)
                 = F(x)

    Then the log-density of the transformed r.v. y is 
    obtained via change of variables and the inverse 
    function theorem as

    \log p(x*) = \log p_x(x) - \sum_k^K \log |\det df_k/dx_k|

    This allows for density estimation of samples from target density 
    p*(x) assuming invertible flow. If it is not possible to evaluate p(x*), 
    we can train parameters of flow F to match p(x*) via divergence 
    minimization. Thus p(\hat{x}) is an approximation of the target density 
    p(x*):

    \log p(x*) \approx \log p_x(F^{-1}(x*)) + \log |\det J_{F^{-1}}(x*)| 

    Parameters
    ----------
    x_flow_inv: List of torch.Tensor
        [x*, x_K, x_{K-1}, ..., x_0]
        Length n_flows, each element has shape (B, x_dim)
    x_dist: distribution Object
        Base distribution of x_0 sample. Typically diagonal-cov. Gaussian
    x_stats: Dict
        Contains parameters of likelihood distribution as tensors.
    log_det_jacobian_inv: 
        Shape (B, n_flows)
    Output
    ----------
    log_pxCz: torch.Tensor
        Log-likelihood under flow model \log p(x* | z).
        Shape (B, x_dim) 
    """


    x_0 = x_flow_inv[-1]
    batch_size = x_0.shape[0]
    # Base distribution is diagonal-covariance Gaussian - TODO: Expand possible base distributions
    # Sum over x_dim
    log_px0Cz = x_dist.log_density(x_0, mu=x_stats['mu'], logvar=x_stats['logvar']).view(batch_size, -1).sum(dim=1)

    # Sum LDJ over flow steps [1,...K]
    log_pxCz = log_px0Cz + log_det_jacobian_inv

    return log_pxCz

def _kl_divergence_q_prior_normal(mu, logvar, per_dim=False):
    """
    Returns KL-divergence between the variational posterior
    $q_{\phi}(z|x)$ and the isotropic Gaussian prior $p(z)$.
    This forms the 'regularization' part of the ELBO.
    
    If the variational posterior is taken to be normal with 
    diagonal covariance. Then:
    $ D_{KL}(q_{\phi(z|x)}||p(z)) = -1/2 * \sum_j (1 + log \sigma_j^2 - \mu_j^2 - \sigma_j^2) $
    """
    
    assert mu.shape == logvar.shape, 'Mean and log-variance must share shape (batch, latent_dim)'
    batch_size, latent_dim = mu.shape
    
    latent_kl = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp()).mean(dim=0)
    total_kl = torch.sum(latent_kl)
    # kl_div = -0.5 * (torch.sum(1 + logvar - mu*mu - torch.exp(logvar)))
    
    if per_dim:
        return total_kl, latent_kl
    else:
        return total_kl


def _kl_divergence_q_prior_uniform(alpha):
    """
    Calculates the KL divergence between a categorical distribution and a
    uniform categorical distribution.
    Parameters
    ----------
    alpha : torch.Tensor
        Parameters of the categorical or gumbel-softmax distribution.
        Shape (N, D)
    """
    EPS = 1e-10
    disc_dim = int(alpha.size()[-1])
    log_dim = torch.Tensor([np.log(disc_dim)])
    log_dim = log_dim.to(alpha.device)
    # Calculate negative entropy of each row
    neg_entropy = torch.sum(alpha * torch.log(alpha + EPS), dim=1)
    # Take mean of negative entropy across batch
    mean_neg_entropy = torch.mean(neg_entropy, dim=0)
    # KL loss of alpha with uniform categorical variable
    kl_loss = log_dim + mean_neg_entropy
    return kl_loss

def _linear_annealing(init, fin, step, annealing_steps):
    """Linear annealing of a parameter. Linearly increase parameter from
       value 'init' to 'fin' over number of iterations specified by
       'annealing_steps'"""
    if annealing_steps == 0:
        return fin
    assert fin > init, 'Final value should be larger than initial'
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def _permute_dims(latent_sample):
    """
    Randomly permutes the sample from
    q(z) (latent_dist) across the batch for each of the latent dimensions (mean
    and log_var).
    Parameters
    ----------
    latent_sample: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        shape : (batch_size, latent_dim).
    """
    perm = torch.zeros_like(latent_sample)
    batch_size, dim_z = perm.size()

    for z in range(dim_z):
        pi = torch.randperm(batch_size).to(latent_sample.device)
        perm[:, z] = latent_sample[pi, z]

    return perm


def _sensitive_generative_matching(latent_stats, generative_factors, sensitive_latent_idx, 
                                   storage=None):
    """
    Supervised loss to match latent factors z with known generative
    factors v. 
    Parameters
    ----------
    latent_factors: torch.Tensor
        sample from the latent dimension using the reparameterisation trick
        or mean of latent distribution. Shape (batch_size, latent_dim)
    generative_factors: torch.Tensor
        true known generative factors of data. Shape (batch_size, latent_dim)
    sensitive_latent_idx: list
        Indices of sensitive latents. Advised to use {0,1,...,d} for d
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


    

# Batch TC specific
# TO-DO: test if mss is better!
def _get_log_pz_qz_prodzi_qzCx(latent_sample, latent_dist, n_data, prior, is_mss=True):
    """
    Get necessary terms to compute decomposed ELBO in (Eq. 2) of [1]
    Parameters
    ----------
    latent_sample: torch.Tensor
        Shape (batch_size, dim)
        sample from distribution defined by encoder network q(z|x)
    latent_dist: (torch.Tensor, torch.Tensor)
        sufficient statistics of encoder distribution. Assumed Gaussian.
        (mu, logvar) -> Shapes (batch_size, dim)
    n_data: int
        size of training dataset
    [1]: https://arxiv.org/pdf/1802.04942.pdf
    """
    batch_size, hidden_dim = latent_sample.shape

    # calculate log q(z|x) with mean, covariance parameterized by encoder network
    log_q_zCx = log_density_gaussian(latent_sample, *latent_dist).sum(dim=1)

    # calculate log p(z) under standard normal prior
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)

    # log_pz = prior.log_density(latent_sample, mu=zeros, logvar=zeros).sum(dim=1)
    log_pz = log_density_gaussian(latent_sample, zeros, zeros).sum(1)
    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)

    # q(z|x) is factorial, but q(z) is not
    # Take product of q(z|x) dimensions before marginalizing x
    # q(z|x) = \prod_i q(z_i | x)
    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    # Marginalize x, then take product to form \prod_i q(z_i)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)

    return log_pz, log_qz, log_prod_qzi, log_q_zCx


# Batch TC specific
# TO-DO: test if mss is better!
def _get_log_pz_qz_prodzi_qzCx_isolate_sensitives(latent_sample, latent_dist, n_data, 
    sensitive_latent_idx, is_mss=True):
    """
    Get necessary terms to compute decomposed ELBO in (Eq. 2) of [1]
    Parameters
    ----------
    latent_sample: torch.Tensor
        Shape (batch_size, dim)
        sample from distribution defined by encoder network q(z|x)
    latent_dist: (torch.Tensor, torch.Tensor)
        sufficient statistics of encoder distribution. Assumed Gaussian.
        (mu, logvar) -> Shapes (batch_size, dim)
    n_data: int
        size of training dataset
    sensitive_latent_idx: list
        Indices of sensitive latents. Advised to use {0,1,...,n} for n
        sensitive latents.

    [1]: https://arxiv.org/pdf/1802.04942.pdf
    """
    batch_size, hidden_dim = latent_sample.shape
    non_sensitive_idx = list(set(range(hidden_dim)) - set(sensitive_latent_idx))
    non_sensitive_idx.sort()

    sensitive_latent_idx = torch.LongTensor(sensitive_latent_idx)
    non_sensitive_idx = torch.LongTensor(non_sensitive_idx)

    # calculate log q(z|x) with mean, covariance parameterized by encoder network
    log_q_zCx_aggregate = log_density_gaussian(latent_sample, *latent_dist)
    log_q_zCx = log_q_zCx_aggregate.sum(dim=1)
    log_q_tCx = log_q_zCx_aggregate[:, sensitive_latent_idx].sum(dim=1)
    log_q_z_minus_tCx = log_q_zCx_aggregate[:, non_sensitive_idx].sum(dim=1)

    # calculate log p(z) under standard normal prior
    # mean and log var is 0
    zeros = torch.zeros_like(latent_sample)
    log_pz_aggregate = log_density_gaussian(latent_sample, zeros, zeros)
    log_pz = log_pz_aggregate.sum(1)
    log_pt = log_pz_aggregate[:, sensitive_latent_idx].sum(1)
    log_pz_minus_t = log_pz_aggregate[:, non_sensitive_idx].sum(1)

    mat_log_qz = matrix_log_density_gaussian(latent_sample, *latent_dist)
    mat_log_qt = mat_log_qz[:,:,sensitive_latent_idx]
    mat_log_qz_minus_t = mat_log_qz[:,:,non_sensitive_idx]

    if is_mss:
        # use stratification
        log_iw_mat = log_importance_weight_matrix(batch_size, n_data).to(latent_sample.device)
        mat_log_qz = mat_log_qz + log_iw_mat.view(batch_size, batch_size, 1)
        mat_log_qt = mat_log_qz[:,:,sensitive_latent_idx]
        mat_log_qz_minus_t = mat_log_qz[:,:,non_sensitive_idx]

    # q(z|x) is factorial, but q(z) is not
    # Take product of q(z|x) dimensions before marginalizing x
    # q(z|x) = \prod_i q(z_i | x)
    log_qz = torch.logsumexp(mat_log_qz.sum(2), dim=1, keepdim=False)
    log_qt = torch.logsumexp(mat_log_qt.sum(2), dim=1, keepdim=False)

    # If all variables are denoted sensitive
    if mat_log_qz_minus_t.nelement() == 0:
        log_q_z_minus_t = torch.zeros(1).to(mat_log_qz_minus_t.device)
    else:
        log_q_z_minus_t = torch.logsumexp(mat_log_qz_minus_t.sum(2), dim=1, keepdim=False)

    # Marginalize x, then take product to form \prod_i q(z_i)
    log_prod_qzi = torch.logsumexp(mat_log_qz, dim=1, keepdim=False).sum(1)
    log_prod_qti = torch.logsumexp(mat_log_qt, dim=1, keepdim=False).sum(1)
    log_prod_qz_minus_ti = torch.logsumexp(mat_log_qz_minus_t, dim=1, keepdim=False).sum(1)

    mi_components = {'log_q_tCx': log_q_tCx,
                     'log_q_z_minus_tCx': log_q_z_minus_tCx,
                     'log_qt': log_qt}

    return log_pz, log_pt, log_pz_minus_t, log_qz, log_q_z_minus_t, log_prod_qzi, log_prod_qti, log_prod_qz_minus_ti, log_q_zCx, mi_components





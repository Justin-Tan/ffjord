""" Using open-source material: 
    @rtiqchen 2018
    @YannDubs 2019
    @riannevdberg 2018 
    @karpathy 2019 """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import network
from utils import math, distributions, initialization

class VAE(nn.Module):
    
    def __init__(self, args, encoder_manual=None, decoder_manual=None):

        """
        Class which defines model and forward pass.
        Parameters
        ----------
        latent_spec : dict
            Specifies latent distribution. For example:
            {'cont': 10, 'disc': [10, 4, 3]} encodes 10 normal variables and
            3 gumbel softmax variables of dimension 10, 4 and 3. A latent spec
            can include both 'cont' and 'disc' or only 'cont'.
            (Not tested with only discrete).

        Specifying both `encoder_manual` and `decoder_manual` overrides the 
        preset encoder/decoder.
        """
        super(VAE, self).__init__()
        
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        self.output_dim = np.prod(args.input_dim)
        self.latent_spec = args.latent_spec
        self.is_discrete = ('discrete' in self.latent_spec.keys())
        self.latent_dim = self.latent_spec['continuous']
        self.flow_output = {'log_det_jacobian': None, 'x_flow': None}

        if not hasattr(args, 'prior') or args.prior == 'normal':
            self.prior = distributions.Normal()
        elif args.prior == 'flow':
            self.prior = distributions.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=args.flow_steps)

        if args.x_dist == 'bernoulli':
            self.x_dist = distributions.Bernoulli()
        elif args.x_dist == 'normal':
            self.x_dist = distributions.Normal()

        if self.is_discrete:
            assert sum(self.latent_spec['discrete']) > 0, 'Must have nonzero number of discrete latent dimensions.'
            print('Using discrete latent factors with specification:', self.latent_spec['discrete'])
            self.latent_dim_discrete = sum([dim for dim in self.latent_spec['discrete']])
            self.n_discrete = len(self.latent_spec['discrete'])
            self.latent_dim += self.latent_dim_discrete  # OK, not 100% consistent

        if args.mlp is True:
            assert args.custom is False, 'Custom option incompatiable with mlp option!'
            encoder = network.MLPEncoder
            decoder = network.MLPDecoder
        else:
            encoder = network.EncoderVAE_conv
            decoder = network.DecoderVAE_conv

        if args.custom is True:
            encoder = network.ToyEncoder
            decoder = network.ToyDecoder

        if encoder_manual is not None and decoder_manual is not None:
            # Manual override
            encoder = encoder_manual
            decoder = decoder_manual

        self.encoder = encoder(input_dim=self.input_dim, latent_spec=self.latent_spec, hidden_dim=self.hidden_dim)
        self.decoder = decoder(input_dim=self.input_dim, latent_dim=self.latent_dim, hidden_dim=self.hidden_dim,
            output_dim=self.output_dim)
        self.reset_parameters()

        print('Using prior:', self.prior)
        print('Using likelihood p(x|z):', self.x_dist)
        print('Using posterior p(z|x): Diagonal-covariance Gaussian')
        
    def reset_parameters(self):
        self.apply(initialization.weights_init)

    def reparameterize(self, latent_stats):
        """
        Combines continuous and discrete latent samples.
        Parameters
        ----------
        latent_stats : dict, torch.Tensor values
            Dict containing at least one of the keys 'continuous' or 'discrete'
            containing the parameters of the latent distributions as torch.Tensor 
            instances.
        """
        mu, logvar = latent_stats['continuous']
        latent_sample = [self.reparameterize_continuous(mu, logvar)]

        if self.is_discrete:
            alphas = latent_stats['discrete']
            discrete_sample = [self.reparameterize_discrete(alpha) for alpha in alphas]
            latent_sample += discrete_sample

        latent_sample = torch.cat(latent_sample, dim=1)

        return latent_sample            
        
        
    def reparameterize_continuous(self, mu, logvar):
        """
        Sample from N(mu(x), Sigma(x)) as 
        z | x ~ mu + Cholesky(Sigma(x)) * eps
        eps ~ N(0,I_n)
        
        The variance is restricted to be diagonal,
        so Cholesky(...) -> sqrt(...)
        Parameters
        ----------
        mu     : torch.Tensor
            Location parameter of Gaussian. (B, latent_dim)

        logvar : torch.Tensor
            Log of variance parameter of Gaussian. (B, latent_dim)

        """
        if self.training:
            sigma = torch.exp(0.5 * logvar)
            epsilon = torch.randn_like(sigma)
            return mu + sigma * epsilon
        else:
            # Reconstruction, return mean
            return mu
        
    def reparameterize_discrete(self, alpha, temperature=0.67):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        Parameters
        ----------
        alpha : torch.Tensor
            Parameters of the gumbel-softmax distribution. Shape (B, latent_dim)
        """
        EPS = 1e-12
        
        
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size()).to(alpha.device)
            gumbel = -torch.log(-torch.log(unif + EPS) + EPS)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + EPS)
            logit = (log_alpha + gumbel) / temperature
            return F.softmax(logit, dim=1)
        else:
            # Reconstruction, return most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).data.cpu(), 1)
            
            one_hot_samples = one_hot_samples.to(alpha.device)
            return one_hot_samples
    
    def forward(self, x):
        latent_stats = self.encoder(x)
        mu, logvar = latent_stats['continuous']
        latent_sample = self.reparameterize(latent_stats)
        x_stats = self.decoder(latent_sample)
        
        return x_stats, latent_sample, latent_stats, self.flow_output


class VAE_ODE(VAE):
    """ Subclass of VAE - replaces decoder with continuous normalizing flow, with
        dynamics determined by a neural network, which is a function of z ~ q(z|x). 
        Performs amortized variational inference; parameters of dynamics network are 
        a function of z.

        Identical encoder logic to standard VAE. Allows density estimation of 
        data x by computing the change in log-density via numerical integration
        by black-box ODE solver. """


    def __init__(self, args):
        super(VAE_ODE, self).__init__(args)
        assert args.flow_type == 'cnf', 'Must toggle CNF option in arguments!'

        dims = self.latent_dim
        self.cnf = build_model_tabular(args, dims)


    def forward(self, x, reverse=False):

        latent_stats = self.encoder(x)
        latent_sample = self.reparameterize(latent_stats)
        x_stats = None

        # Compute change in log-prob via numerical integration
        zero = torch.zeros(latent_sample.shape[0], 1).to(latent_sample)
        x, delta_logp = self.cnf(latent_sample, zero)

        self.flow_output['log_det_jacobian'] = -delta_logp.view(-1)
        self.flow_output['x_flow'] = x
        self.flow_output['z_0'] = latent_sample

        return x_stats, latent_sample, latent_stats, self.flow_output
    

class realNVP_VAE(VAE):
    """ Subclass of VAE - implements invertible normalizing flows in the decoder.
        Identical encoder logic to standard VAE. Allows density estimation of 
        data x = T(u) by computing log p(T^{-1}(x)) + log |det J_{T^{-1}}(x)|}. """

    def __init__(self, args):
        super(realNVP_VAE, self).__init__(args)
        
        assert args.use_flow is True, 'Must toggle use_flow option in arguments!'

        # TODO: Expand possible invertible flows 
        flow = distributions.InvertibleAffineFlow
        self.n_flows = args.flow_steps

        # Aggregate parameters from each transformation in the flow
        parity = lambda n: True if n%2==0 else False
        for k in range(self.n_flows):
            flow_k = flow(input_dim=self.input_dim, parity=parity(k), hidden_dim=args.flow_hidden_dim)
            self.add_module('flow_{}'.format(str(k)), flow_k)


    def forward(self, x, sample=False):
        latent_stats = self.encoder(x)
        latent_sample = self.reparameterize(latent_stats)

        # Parameters of base distribution - diagonal covariance Gaussian
        x_stats = self.decoder(latent_sample)

        if sample is True:
            # Return reconstructed sample from target density
            flow_output = self.forward_flow(x_stats)
            return x_stats, latent_sample, latent_stats, flow_output
        else:
            # Invert flow to fit flow-based model to target density
            # Note reversed order of flow, LDJ in backward output
            inv_flow_output = self.backward_flow(x)
            x0 = inv_flow_output['x_flow_inv'][-1]
            return x_stats, latent_sample, latent_stats, inv_flow_output


    def forward_flow(self, x_stats):
        """ Sample from target density by passing x0 sampled from
            base distribution through forward flow, x = F(x_0). 
            Used for sampling from trained model. """

        # Sample x_0
        x_0 = self.reparameterize_continuous(mu=x_stats['mu'], logvar=x_stats['logvar'])
        batch_size = x_0.shape[0]
        x_flow = [x_0]  # Sequence of residual flows. \hat{x} = x_flow[-1]

        log_det_jacobian = torch.zeros(batch_size).type_as(x.data)

        for k in range(self.n_flows):
            flow_k = getattr(self, 'flow_{}'.format(str(k)))
            x_k, log_det_jacobian_k = flow_k.forward(x_flow[k])
            x_flow.append(x_k)
            log_det_jacobian += log_det_jacobian_k

        # Final approximation of target sample
        x_K = x_flow[-1]

        flow_output = {'log_det_jacobian': log_det_jacobian, 'x_flow': x_flow}

        return flow_output 


    def backward_flow(self, x):
        """ Recover base x0 ~ N(\mu, \Sigma) by inverting 
            flow transformations. x0 = F^{-1}(x) Used for density 
            evaluation when computing likelihood term in VAE loss. """

        x_flow_inv = [x]
        batch_size = x.shape[0]
        log_det_jacobian_inv = torch.zeros(batch_size).type_as(x.data)

        # Sequence T^{-1}(x) = u ==> x -> x_{K-1} -> ... -> x_1 -> u
        for k in range(self.n_flows)[::-1]:  # reverse order
            idx = self.n_flows - (k+1)
            flow_k = getattr(self, 'flow_{}'.format(str(k)))
            x_k, log_det_jacobian_k = flow_k.invert(x_flow_inv[idx])
            x_flow_inv.append(x_k)
            log_det_jacobian_inv += log_det_jacobian_k

        # Sample u = x_0 = T^{-1}(x) from base distribution 
        x_0 = x_flow_inv[-1] 

        inv_flow_output = {'log_det_jacobian_inv': log_det_jacobian_inv, 'x_flow_inv': x_flow_inv}

        return inv_flow_output

class PlanarVAE(VAE):
    """ Subclass of VAE - implements planar flows in the encoder.
        Identical decoder logic to standard VAE, changes encoder
        to implement z_0 -> z_1 -> ... -> z_K flow. """

    def __init__(self, args):
        super(PlanarVAE, self).__init__(args)
        
        assert args.use_flow is True, 'Must toggle use_flow option in arguments!'

        flow = distributions.PlanarFlow
        self.n_flows = args.flow_steps

        # Amortized flow parameters
        self.amor_u = nn.Linear(self.hidden_dim, self.n_flows * self.latent_dim)
        self.amor_w = nn.Linear(self.hidden_dim, self.n_flows * self.latent_dim)
        self.amor_b = nn.Linear(self.hidden_dim, self.n_flows)

        # Normalizing flow layers
        for k in range(self.n_flows):
            flow_k = flow(dim=self.latent_dim)
            self.add_module('flow_{}'.format(str(k)), flow_k)

        self.planar_flow = flow(dim=self.output_dim)


    def forward(self, x):
        """
        Normalizing flow in decoder implements more flexible likelihood term to handle 
        non-Gaussian densities.
        """
        batch_size = x.shape[0]

        latent_stats = self.encoder(x)
        latent_sample = self.reparameterize(latent_stats)
        z_0 = latent_sample

        z_flow = [z_0]  # Sequence of residual flows. \hat{x} = x_flow[-1]
        h = latent_stats['hidden']  # activation before projection into mu, logvar

        # return amortized u/w/b for all flows
        u = self.amor_u(h).view(batch_size, self.n_flows, self.latent_dim, 1)
        w = self.amor_w(h).view(batch_size, self.n_flows, 1, self.latent_dim)
        b = self.amor_b(h).view(batch_size, self.n_flows, 1, 1)

        log_det_jacobian = torch.zeros([batch_size, self.n_flows], requires_grad=True).type_as(x.data)

        for k in range(self.n_flows):
            flow_k = getattr(self, 'flow_{}'.format(str(k)))
            z_k, log_det_jacobian_k = flow_k(z_flow[k], u[:,k,:,:], w[:,k,:,:], b[:,k,:,:])
            # x_k, log_det_jacobian_k = self.planar_flow(x_flow[k], u[:,k,:,:], w[:,k,:,:], b[:,k,:,:])
            z_flow.append(z_k)
            log_det_jacobian[:,k] = log_det_jacobian_k
            # log_det_jacobian += log_det_jacobian_k

        # Final approximation of target density
        z_K = z_flow[-1]

        # Parameters of output distribution
        x_stats = self.decoder(z_K)

        self.flow_output = {'log_det_jacobian': log_det_jacobian, 'z_flow': z_flow}

        return x_stats, latent_sample, latent_stats, self.flow_output

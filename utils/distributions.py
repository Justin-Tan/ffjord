import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import math
from models import network

EPS = 1e-8

class Normal(nn.Module):

    """
    Normal distribution with diagonal covariance
    """
    def __init__(self, mu=0., logvar=1.):
        super(Normal, self).__init__()

        self.mu = torch.Tensor([mu])
        self.logvar = torch.Tensor([logvar])

    def sample(self, mu, logvar):
        """
        Sample from N(mu(x), Sigma(x)) as 
        z ~ mu + Cholesky(Sigma(x)) * eps
        eps ~ N(0,I_n)
        
        The variance is restricted to be diagonal,
        so Cholesky(...) -> sqrt(...)
        """
        sigma_sqrt = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(sigma_sqrt)
        return mu + sigma_sqrt * epsilon

    def log_density(self, x, mu, logvar):
        """
        First argument of params is location parameter,
        second argument is scale parameter
        """
        return math.log_density_gaussian(x, mu, logvar)

    def NLL(self, params, sample_params=None):
        """
        Analytically computes negative log-likelihood 
        E_N(mu_2, var_2) [- log N(mu_1, var_1)]
        If mu_2, and var_2 are not provided, defaults to entropy.
        """
        mu, logvar = params

        if sample_params is not None:
            sample_mu, sample_logvar = sample_params
        else:
            sample_mu, sample_logvar = mu, logvar

        c = self.normalization.type_as(sample_mu.data)
        nll = logsigma.mul(-2).exp() * (sample_mu - mu).pow(2) \
            + torch.exp(sample_logsigma.mul(2) - logsigma.mul(2)) + 2 * logsigma + c
        return nll.mul(0.5)

class FactorialNormalizingFlow(nn.Module):

    def __init__(self, dim, nsteps=32, flow_type='planar'):
        super(FactorialNormalizingFlow, self).__init__()
        self.dim = dim
        self.nsteps = nsteps
        self.scale = nn.Parameter(torch.Tensor(self.nsteps, self.dim))
        self.weight = nn.Parameter(torch.Tensor(self.nsteps, self.dim))
        self.bias = nn.Parameter(torch.Tensor(self.nsteps, self.dim))
        self.reset_parameters()

        if (flow_type=='planar'):
            self.flow = self.planar_flow
        elif (flow_type=='radial'):
            self.flow = self.radial_flow

    def reset_parameters(self):
        self.scale.data.normal_(0, 0.02)
        self.weight.data.normal_(0, 0.02)
        self.bias.data.normal_(0, 0.02)

    def planar_flow(self, x, logdetgrad=None):

        for i in range(self.nsteps):
            u = self.scale[i][None]
            w = self.weight[i][None]
            b = self.bias[i][None]
            act = torch.tanh(x * w + b)
            x = x + u * act

            if logdetgrad is not None:
                logdetgrad = logdetgrad + torch.log(torch.abs(1 + u * (1 - act.pow(2)) * w) + EPS)

        if logdetgrad is not None:
            return x, logdetgrad
        
        return x

    def sample(self, batch_size):
        # Sample from standard normal,
        # pass through planar flow
        # TODO: try radial flow

        z_0 = torch.randn([batch_size, self.dim])
        z_K = self.planar_flow(z_0)
        return z_K

    def log_density(self, y, params=None, **kwargs):
        assert(y.size(1) == self.dim)
        x = y
        logdetgrad = torch.zeros(y.size(), requires_grad=True).type_as(y.data)
        x, logdetgrad = self.planar_flow(x, logdetgrad)

        zeros = torch.zeros_like(x)
        logpx = math.log_density_gaussian(x, mu=zeros, logvar=zeros)
        logpy = logpx - logdetgrad 
        return logpy

class Bernoulli(nn.Module):
    """
    Bernoulli distribution. Probability given by sigmoid of input parameter.
    For natural image data each pixel is modelled as independent Bernoulli
    """
    def __init__(self, theta=0.5):
        super(Bernoulli, self).__init__()

        self.theta = torch.Tensor([theta])

    def sample(self, theta):
        """
        """
        raise NotImplementedError


    def log_density(self, x, params=None):
        """ x, params \in [0,1] """
        log_px = F.binary_cross_entropy(params, x, reduction="sum")
        return log_px


class PlanarFlow(nn.Module):
    """
    For use in VAE encoder. Augments expressiveness of approximate posterior.
    """
    def __init__(self, dim, flow_type='planar'):
        super(PlanarFlow, self).__init__()

        self.dim = dim
        self.activation = nn.Tanh()
        self.softplus = nn.Softplus()

    def grad_activation(self, x):
        """ Derivative of tanh """

        return 1 - self.activation(x) ** 2

    def forward(self, zk, u, w, b):
        """
        Forward pass. Assumes amortized u, w and b. Conditions on diagonals of u and w for invertibility
        will be be satisfied inside this function. Computes the following transformation:
        z' = z + u h( w^T z + b)
        or actually
        z'^T = z^T + h(z^T w + b)u^T
        Assumes the following input shapes:
        shape u = (batch_size, z_size, 1)
        shape w = (batch_size, 1, z_size)
        shape b = (batch_size, 1, 1)
        shape z = (batch_size, z_size).
        """
        assert zk.shape[1] == self.dim, 'Dimensions must match supplied arguments!'
        zk = zk.unsqueeze(2)

        # reparameterize u such that the flow becomes invertible (see appendix paper)
        uw = torch.bmm(w, u)
        m_uw = -1. + self.softplus(uw)
        w_norm_sq = torch.sum(w ** 2, dim=2, keepdim=True)
        u_hat = u + ((m_uw - uw) * w.transpose(2, 1) / w_norm_sq)

        # compute flow with u_hat
        wzb = torch.bmm(w, zk) + b
        z = zk + u_hat * self.activation(wzb)
        z = z.squeeze(2)

        # compute logdetJ
        psi = w * self.grad_activation(wzb)
        log_det_jacobian = torch.log(torch.abs(1 + torch.bmm(psi, u_hat))  + EPS)
        log_det_jacobian = log_det_jacobian.squeeze(2).squeeze(1)  # Shape [batch_size]

        return z, log_det_jacobian

    def log_density(z_0, latent_stats, log_det_jacobian):
        """
        Let z_K = f_K \circ f_{K-1} \circ ... \circ f_1 (z_0)
        Then the log-density of the transformed r.v. y is
        log p(z_K) = log p(z_0) - \sum_k^K \log \det |df_k/dz_k|
        via change of variables.

        Parameters
        ----------
        z_0 : torch.Tensor
            Input data. Shape (B, z_dim)
        latent_stats: Dict
            Contains parameters of posterior distribution as tensors.
        log_det_jacobian:
            Shape (B, n_flows)
        """

        batch_size = z.shape[0]

        # Base distribution is diagonal-covariance Gaussian
        log_qz0Cx = math.log_density_gaussian(z_0, mu=latent_stats['mu'], logvar=latent_stats['logvar']).view(batch_size, -1).sum(1)

        # Sum LDJ over flow steps [1,...K]
        log_qzKCx = log_qz0Cx - log_det_jacobian.sum(dim=1)  # Shape (B)

        return log_qzKCx

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """
    
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)        
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)
    

class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """
        
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        
        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)
        
        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings
        
        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.
        
    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)
        
        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        
        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), self.nin-1, size=self.hidden_sizes[l])
        
        # construct the mask matrices
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])
        
        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)
        
        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x):
        return self.net(x)

class ARMLP(nn.Module):
    """ a 4-layer auto-regressive MLP, wrapper around MADE net """

    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.net = MADE(input_dim, [hidden_dim, hidden_dim, hidden_dim], output_dim, num_masks=1, natural_ordering=True)
        
    def forward(self, x):
        return self.net(x)

class MAF(nn.Module):
    """ Masked Autoregressive Flow that uses a MADE-style network for fast forward """
    
    def __init__(self, input_dim, parity, hidden_dim=64, net_class=ARMLP):
        super().__init__()
        self.dim = input_dim
        self.net = net_class(input_dim, input_dim*2, hidden_dim)
        self.parity = parity

    def forward(self, x):
        # here we see that we are evaluating all of z in parallel, so density estimation will be fast
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        # reverse order, so if we stack MAFs correct things happen
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def invert(self, z):
        # we have to decode the x one at a time, sequentially
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0)).type_as(z.data)
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone()) # clone to avoid in-place op errors if using IAF
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det


class InvertibleAffineFlow(nn.Module):
    """ 
    Implements real-NVP affine autoregressive flow [1] as 
    location-scale transform. 

    Parition input z = [z1 | z2], then applies autoregressive
    transformation z' = f(z), wehre f:
    z1' = z1
    z2' = exp(s(z1)) * z2 + t(z1)

    [1] Density estimation using Real-NVP, Dinh et. al. 2016
        arXiv:1605.08803
    """

    def __init__(self, input_dim, parity=False, hidden_dim=64):
        super(InvertibleAffineFlow, self).__init__()

        self.input_dim = input_dim
        self.parity = parity
        self.hidden_dim = hidden_dim

        self.net = network.NVP_net

        self.s_psi = self.net(input_dim=self.input_dim//2, output_dim=self.input_dim//2, hidden_dim=self.hidden_dim)
        self.t_psi = self.net(input_dim=self.input_dim//2, output_dim=self.input_dim//2, hidden_dim=self.hidden_dim)

    def forward(self, x):
        x0, x1 = x[:,::2], x[:,1::2]
        if self.parity is True:  # Exchange partitions on successive transformations
            x0, x1 = x1, x0

        s_x0, t_x0 = self.s_psi(x0), self.t_psi(x0)
        z0 = x0  # Identity
        z1 = torch.exp(s_x0) * x1 + t_x0  # location-scale transform

        if self.parity is True:
            z0, z1 = z1, z0

        z = torch.cat([z0, z1], dim=1)
        log_det_jacobian = torch.sum(s_x0, dim=1)
        
        return z, log_det_jacobian


    def invert(self, z):
        z0, z1 = z[:,::2], z[:,1::2]
        if self.parity is True:
            z0, z1 = z1, z0

        s_z0, t_z0 = self.s_psi(z0), self.t_psi(z0)
        x0 = z0  # Identity
        x1 = (z1 - t_z0) * torch.exp(-s_z0)  # Inverse location-scale transform

        if self.parity is True:
            x0, x1 = x1, x0

        x = torch.cat([x0, x1], dim=1)

        # |det J_{T^{-1}}(x)| = |det(J_T(u))|^{-1}
        log_det_jacobian = -torch.sum(s_z0, dim=1)

        return x, log_det_jacobian


class DiscreteFlowModel(nn.Module):
    """ Allows density estimation of  data x = T(u) by computing 
        p(T^{-1}(x)) + log |det J_{T^{-1}}(x)|}. 
        Composes arbitrary flows. """

    def __init__(self, input_dim, hidden_dim, base_dist=None, n_flows=8,
                 flow=InvertibleAffineFlow):
        super(DiscreteFlowModel, self).__init__()
        
        self.n_flows = n_flows
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.base_dist = base_dist
        
        BN_flow = BatchNormFlow
        parity = lambda n: True if n%2==0 else False
        
        # Aggregate parameters from each transformation in the flow
        for k in range(self.n_flows):
            flow_k = flow(input_dim=self.input_dim, parity=parity(k), hidden_dim=self.hidden_dim)
            BN_k = BN_flow(input_dim=self.input_dim)
            self.add_module('flow_{}'.format(str(k)), flow_k)
            self.add_module('BN_{}'.format(str(k)), BN_k)


    def forward(self, x_0):
        """ Sample from target density by passing x0 sampled from
            base distribution through forward flow, x = F(x_0). 
            Used for sampling from trained model. """

        batch_size = x_0.shape[0]
        x_flow = [x_0]  # Sequence of residual flows. \hat{x} = x_flow[-1]

        log_det_jacobian = torch.zeros(batch_size).to(x_0)

        for k in range(self.n_flows):
            
            x_k = x_flow[-1]
            flow_k = getattr(self, 'flow_{}'.format(str(k)))
            x_k, log_det_jacobian_k = flow_k.forward(x_k)
            
            # Don't apply batch norm after final forward flow T_{K-1}
            if k < self.n_flows - 1:
                BN_k = getattr(self, 'BN_{}'.format(str(k)))
                x_k, log_det_jacobian_BN_k = BN_k.forward(x_k)
                log_det_jacobian += log_det_jacobian_BN_k
                
            x_flow.append(x_k)
            
            log_det_jacobian += log_det_jacobian_k


        # Final approximation of target sample
        x_K = x_flow[-1]

        flow_output = {'log_det_jacobian': log_det_jacobian, 'x_flow': x_flow}

        return flow_output 


    def backward(self, x):
        """ Recover base x0 ~ N(\mu, \Sigma) by inverting 
            flow transformations. x0 = F^{-1}(x) Used for density 
            evaluation when computing likelihood term in VAE loss. """

        x_flow_inv = [x]
        batch_size = x.shape[0]
        log_det_jacobian_inv = torch.zeros(batch_size).to(x)

        # Sequence T^{-1}(x) = u ==> x -> x_{K-1} -> ... -> x_1 -> u
        for k in range(self.n_flows)[::-1]:  # reverse order
            
            x_k = x_flow_inv[-1]
            
            # Don't apply batch norm before transform T^{-1}_{K_1}
            if k < self.n_flows - 1:
                BN_k = getattr(self, 'BN_{}'.format(str(k)))
                x_k, log_det_jacobian_BN_k = BN_k.invert(x_k)
                log_det_jacobian_inv += log_det_jacobian_BN_k

            flow_k = getattr(self, 'flow_{}'.format(str(k)))
            x_k, log_det_jacobian_k = flow_k.invert(x_k)
        
            x_flow_inv.append(x_k)
            
            log_det_jacobian_inv += log_det_jacobian_k

        inv_flow_output = {'log_det_jacobian_inv': log_det_jacobian_inv, 'x_flow_inv': x_flow_inv}

        return inv_flow_output
    
    def log_density(self, x):
        
        inv_flow_output = self.backward(x)
        
        log_det_jacobian_inv = inv_flow_output['log_det_jacobian_inv']
        x_flow_inv = inv_flow_output['x_flow_inv']
        
        x_0 = x_flow_inv[-1]
        log_px_0 = self.base_dist.log_prob(x_0.cpu()).cuda()
        log_px = log_px_0 + log_det_jacobian_inv
        
        return log_px
    
    def sample(self, n_samples):
        
        x_0 = self.base_dist.sample(torch.Size([n_samples])).cuda()
        
        flow_output = self.forward(x_0)
        log_det_jacobian = flow_output['log_det_jacobian']
        x_flow = flow_output['x_flow']
        x_K = x_flow[-1]
        
        log_px_0 = self.base_dist.log_prob(x_0.cpu()).cuda()
        
        log_px_K = log_px_0 - log_det_jacobian
        
        return x_K, log_px_K
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

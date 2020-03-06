import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import math

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
        logpy = logpx + logdetgrad 
        return logpy

class Bernoulli(nn.Module):

    """
    Bernoulli distribution. Probability given by sigmoid of input parameter.
    For natural image data each pixel is modelled as independent Bernoulli
    """
    def __init__(self, theta=0.5):
        super(Bernoulli, self).__init__()

        self.theta = torch.Tensor([theta])

    def sample(self, mu, logvar):
        """
        """
        raise NotImplementedError


    def log_density(self, x, params=None):
        log_px = F.binary_cross_entropy(params, x, reduction="sum")
        return log_px



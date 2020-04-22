
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Parameter


class BatchNormFlow(nn.Module):
    """ Implements forward and inverse batch
        norm passes, as well as log determinant of
        Jacobian of forward and inverse passes """
    
    def __init__(self, input_dim, momentum=0.05, eps=1e-5):
        super(BatchNormFlow, self).__init__()
        
        self.log_gamma = nn.Parameter(torch.zeros(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))
        self.momentum = momentum
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(input_dim))
        self.register_buffer('running_var', torch.ones(input_dim))
        

    def forward(self, x, reverse=False):
        if reverse:
            return self._invert(x)
        else:
            return self._forward(x)

    def _forward(self, u, logpu=None):
        """
        Use recorded running mean/var to invert BN applied during
        inverse pass.
        """
        mu, var = self.running_mean, self.running_var
        
        x = (u - self.beta) * torch.exp(-self.log_gamma) * torch.sqrt(var + self.eps) + mu
        log_det_jacobian_inv = -torch.sum(self.log_gamma - 0.5 * torch.log(var + self.eps))

        return x, log_det_jacobian_inv
   
        
    def _invert(self, x, logpx=None):
        """
        Apply BN using minibatch statistics, update running mean/var.
        """
        batch_mu = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        
        self.running_mean.mul_(self.momentum)
        self.running_var.mul_(self.momentum)

        self.running_mean.add_(batch_mu.data * (1 - self.momentum))
        self.running_var.add_(batch_var.data * (1 - self.momentum))
        
        x = torch.exp(self.log_gamma) * (x - batch_mu) * 1. / torch.sqrt(batch_var + self.eps) + self.beta
        log_det_jacobian = torch.sum(self.log_gamma -0.5 * torch.log(batch_var + self.eps))
        
        return x, log_det_jacobian


class MovingBatchNormNd(nn.Module):
    def __init__(self, num_features, eps=1e-4, decay=0.1, bn_lag=0., affine=True):
        super(MovingBatchNormNd, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.decay = decay
        self.bn_lag = bn_lag
        self.register_buffer('step', torch.zeros(1))
        if self.affine:
            self.bn_weight = Parameter(torch.Tensor(num_features))
            self.bn_bias = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.reset_parameters()

    @property
    def shape(self):
        raise NotImplementedError

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        if self.affine:
            self.bn_weight.data.zero_()
            self.bn_bias.data.zero_()

    def forward(self, x, logpx=None, reverse=False):
        if reverse:
            return self._reverse(x, logpx)
        else:
            return self._forward(x, logpx)

    def _forward(self, x, logpx=None):
        c = x.size(1)
        used_mean = self.running_mean.clone().detach()
        used_var = self.running_var.clone().detach()

        if self.training:
            # compute batch statistics
            x_t = x.transpose(0, 1).contiguous().view(c, -1)
            batch_mean = torch.mean(x_t, dim=1)
            batch_var = torch.var(x_t, dim=1)

            # moving average
            if self.bn_lag > 0:
                used_mean = batch_mean - (1 - self.bn_lag) * (batch_mean - used_mean.detach())
                used_mean /= (1. - self.bn_lag**(self.step[0] + 1))
                used_var = batch_var - (1 - self.bn_lag) * (batch_var - used_var.detach())
                used_var /= (1. - self.bn_lag**(self.step[0] + 1))

            # update running estimates
            self.running_mean -= self.decay * (self.running_mean - batch_mean.data)
            self.running_var -= self.decay * (self.running_var - batch_var.data)
            self.step += 1

        # perform normalization
        used_mean = used_mean.view(*self.shape).expand_as(x)
        used_var = used_var.view(*self.shape).expand_as(x)

        y = (x - used_mean) * torch.exp(-0.5 * torch.log(used_var + self.eps))

        if self.affine:
            weight = self.bn_weight.view(*self.shape).expand_as(x)
            bias = self.bn_bias.view(*self.shape).expand_as(x)
            y = y * torch.exp(weight) + bias

        if logpx is None:
            return y
        else:
            return y, logpx - self._logdetgrad(x, used_var).view(x.size(0), -1).sum(1, keepdim=True)

    def _reverse(self, y, logpy=None):
        used_mean = self.running_mean
        used_var = self.running_var

        if self.affine:
            weight = self.bn_weight.view(*self.shape).expand_as(y)
            bias = self.bn_bias.view(*self.shape).expand_as(y)
            y = (y - bias) * torch.exp(-weight)

        used_mean = used_mean.view(*self.shape).expand_as(y)
        used_var = used_var.view(*self.shape).expand_as(y)
        x = y * torch.exp(0.5 * torch.log(used_var + self.eps)) + used_mean

        if logpy is None:
            return x
        else:
            return x, logpy + self._logdetgrad(x, used_var).view(x.size(0), -1).sum(1, keepdim=True)

    def _logdetgrad(self, x, used_var):
        logdetgrad = -0.5 * torch.log(used_var + self.eps)
        if self.affine:
            weight = self.bn_weight.view(*self.shape).expand(*x.size())
            logdetgrad += weight
        return logdetgrad

    def __repr__(self):
        return (
            '{name}({num_features}, eps={eps}, decay={decay}, bn_lag={bn_lag},'
            ' affine={affine})'.format(name=self.__class__.__name__, **self.__dict__)
        )


def stable_var(x, mean=None, dim=1):
    if mean is None:
        mean = x.mean(dim, keepdim=True)
    mean = mean.view(-1, 1)
    res = torch.pow(x - mean, 2)
    max_sqr = torch.max(res, dim, keepdim=True)[0]
    var = torch.mean(res / max_sqr, 1, keepdim=True) * max_sqr
    var = var.view(-1)
    # change nan to zero
    var[var != var] = 0
    return var


class MovingBatchNorm1d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1]


class MovingBatchNorm2d(MovingBatchNormNd):
    @property
    def shape(self):
        return [1, -1, 1, 1]
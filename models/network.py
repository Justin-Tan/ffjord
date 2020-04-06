""" Using material from: 
    @YannDubs 2019
    @sksq96   2019 """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
=====================
Encoders / Decoders
=====================
"""

class EncoderVAE_conv(nn.Module):
    def __init__(self, input_dim, activation='relu', latent_spec=None, hidden_dim=256):
        """ 
        Gaussian encoder $q_{\phi}(z|x)$ with convolutional 
        architecture proposed in [1].
        
        The mean and log-variance of each latent dimension is 
        parameterized by the encoder. $z$ can be later sampled 
        using the reparameterization trick.
        
        [1] Locatello et. al., "Challenging Common Assumptions
        in the Unsupervised Learning of Disentangled 
        Representations", arXiv:1811.12359 (2018).
        """
        
        super(EncoderVAE_conv, self).__init__()
        
        self.latent_dim_continuous = latent_spec['continuous']
        self.input_dim = input_dim
        im_channels = self.input_dim[0]
        kernel_dim = 4
        hidden_channels = 64
        n_ch1 = 32
        n_ch2 = 64
        cnn_kwargs = dict(stride=2, padding=1)
        out_conv_shape = (hidden_channels, kernel_dim, kernel_dim)
        # (leaky_relu, relu, elu)
        self.activation = getattr(F, activation)
        
        self.conv1 = nn.Conv2d(im_channels, n_ch1, kernel_dim, **cnn_kwargs)
        self.conv2 = nn.Conv2d(n_ch1, n_ch1, kernel_dim, **cnn_kwargs)
        self.conv3 = nn.Conv2d(n_ch1, n_ch2, kernel_dim, **cnn_kwargs)
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            self.conv4 = nn.Conv2d(n_ch2, n_ch2, kernel_dim, **cnn_kwargs)
            
        self.dense1 = nn.Linear(np.product(out_conv_shape), hidden_dim)    
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.dense_mu = nn.Linear(hidden_dim, self.latent_dim_continuous)
        self.dense_logvar = nn.Linear(hidden_dim, self.latent_dim_continuous)
        self.is_discrete = ('discrete' in latent_spec.keys())


        if self.is_discrete is True:
            # Specify parameters of categorical distribution
            self.discrete_latents = latent_spec['discrete']
            dense_alphas = [nn.Linear(hidden_dim, alpha_dim) for alpha_dim in self.discrete_latents]
            self.dense_alphas = nn.ModuleList(dense_alphas)
            
    def forward(self, x):
        
        # Holds parameters of latent distribution
        # Divided into continuous and discrete dims
        latent_dist = {'continuous': None}

        batch_size = x.size(0)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            x = self.activation(self.conv4(x))
            
        x = x.view((batch_size, -1))
        x = self.activation(self.dense1(x))
        # x = activation(self.dense2(x))
        
        mu = self.dense_mu(x)
        logvar = self.dense_logvar(x)
        latent_dist['continuous'] = [mu, logvar]
        latent_dist['hidden'] = x


        if self.is_discrete:
            latent_dist['discrete'] = [F.softmax(dense_alpha(x), dim=1) for dense_alpha in self.dense_alphas]
        
        return latent_dist

class DecoderVAE_conv(nn.Module):
    def __init__(self, input_dim, latent_dim=10, activation='relu', **kwargs):
        """ 
        Gaussian decoder $p_{\theta}(x|z) $ with convolutional 
        architecture used in [1].
        
        The mean and log-variance of the reconstruction $\hat{x}$ 
        is again parameterized by the decoder.
        
        [1] Locatello et. al., "Challenging Common Assumptions
        in the Unsupervised Learning of Disentangled 
        Representations", arXiv:1811.12359 (2018).
        """
        
        super(DecoderVAE_conv, self).__init__()
        
        self.input_dim = input_dim
        im_channels = self.input_dim[0]
        kernel_dim = 4
        hidden_dim = 256
        hidden_channels = 64
        n_ch1 = 64
        n_ch2 = 32
        cnn_kwargs = dict(stride=2, padding=1)
        # (leaky_relu, relu, elu)
        self.activation = getattr(F, activation)
        self.reshape = (hidden_channels, kernel_dim, kernel_dim)
        
        self.dense1 = nn.Linear(latent_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, np.product((self.reshape)))
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            self.upconv_i = nn.ConvTranspose2d(n_ch1, n_ch1, kernel_dim, **cnn_kwargs)
        
        self.upconv1 = nn.ConvTranspose2d(n_ch1, n_ch2, kernel_dim, **cnn_kwargs)
        self.upconv2 = nn.ConvTranspose2d(n_ch2, n_ch2, kernel_dim, **cnn_kwargs)
        self.upconv3 = nn.ConvTranspose2d(n_ch2, im_channels, kernel_dim, **cnn_kwargs)
        
    def forward(self, z):
        
        batch_size = z.size(0)
        x = self.activation(self.dense1(z))
        x = self.activation(self.dense2(x))
        x = x.view((batch_size, *self.reshape))
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            x = self.activation(self.upconv_i(x))
            
        x = self.activation(self.upconv1(x))
        x = self.activation(self.upconv2(x))
        logits = self.upconv3(x)
        
        # Implicitly assume output is Bernoulli distributed - bad?
        out = torch.sigmoid(logits)
        
        return out


class MLPEncoder(nn.Module):
    """ For image data.
    """
    def __init__(self, input_dim, latent_spec, **kwargs):
        super(MLPEncoder, self).__init__()

        hidden_dim = 256
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim_continuous = latent_spec['continuous']

        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, hidden_dim)

        self.dense_mu = nn.Linear(hidden_dim, self.latent_dim_continuous)
        self.dense_logvar = nn.Linear(hidden_dim, self.latent_dim_continuous)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        h = x.view(-1, self.input_dim[1] * self.input_dim[2])
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.fc3(h)
        z = h.view(x.size(0), self.hidden_dim)
        latent_dist = {}

        mu = self.dense_mu(z)
        logvar = self.dense_logvar(z)

        latent_dist['continuous'] = [mu, logvar]
        latent_dist['hidden'] = z

        return latent_dist



class MLPDecoder(nn.Module):
    """ For image data.
    """
    def __init__(self, input_dim, latent_dim=10, output_dim=4096, **kwargs):
        super(MLPDecoder, self).__init__()

        self.input_dim = input_dim
        self.fc1 = nn.Linear(latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, output_dim)

        self.act = nn.ReLU(inplace=True)  # or nn.Tanh

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 1024), nn.Tanh(),
            nn.Linear(1024, 1024), nn.Tanh(),
            nn.Linear(1024, 1024), nn.Tanh(),
            nn.Linear(1024, 4096)
        )

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        # h = self.act(self.fc3(h))
        h = self.fc4(h)

        logits = h.view(z.size(0), self.input_dim[0], self.input_dim[1], self.input_dim[2])
        reconstruction = torch.sigmoid(logits)

        return reconstruction


class ToyEncoder(nn.Module):
    def __init__(self, latent_spec, input_dim=2, hidden_dim=256, activation='elu', **kwargs):
        super(ToyEncoder, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_spec['continuous']
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.dense_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.dense_logvar = nn.Linear(hidden_dim, self.latent_dim)
        
        # self.act = nn.ReLU(inplace=True)
        self.act = getattr(F, activation)
        self.ht = nn.Hardtanh(min_val=-5, max_val=5)


    def forward(self, x):

        h = x.view(-1, self.input_dim)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        # h = self.act(self.fc4(h))
        latent_dist = {}

        mu = self.dense_mu(h)
        logvar = self.dense_logvar(h)
        logvar = torch.clamp(logvar, min=-5, max=5)
            # post_logvar = self.ht(post_logvar)

        latent_dist['continuous'] = [mu, logvar]
        latent_dist['hidden'] = h

        return latent_dist


class ToyDecoder(nn.Module):
    def __init__(self, latent_dim=10, output_dim=2, hidden_dim=256, activation='elu', **kwargs):
        super(ToyDecoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)

        self.dense_post_mu = nn.Linear(hidden_dim, self.output_dim)
        self.dense_post_logvar = nn.Linear(hidden_dim, self.output_dim)

        # self.act = nn.ReLU(inplace=True)  # or nn.Tanh
        self.act = getattr(F, activation)
        self.ht = nn.Hardtanh(min_val=-5, max_val=5)

    def forward(self, z):
        h = z.view(z.size(0), -1)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        # h = self.act(self.fc4(h))

        post_mu = self.dense_post_mu(h)
        post_logvar = self.dense_post_logvar(h)
        post_logvar = torch.clamp(post_logvar, min=-5, max=5)
        # post_logvar = self.ht(post_logvar)

        return dict(mu=post_mu, logvar=post_logvar, hidden=h)


class NVP_net(nn.Module):
    """ Network for use for transforms in real-NVP """

    def __init__(self, input_dim, output_dim, hidden_dim=128, activation='leaky_relu', scale=False):
        super(NVP_net, self).__init__()

        self.input_dim = input_dim
        self.scale = scale
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.scale_act = nn.Tanh()
        self.tanh_scale = nn.Parameter(torch.ones(1), requires_grad=True)

        # self.act = nn.LeakyReLU(0.2, inplace=True)  # nn.ReLU(inplace=True)
        self.act = getattr(F, activation)

    def forward(self, x):

        h = x.view(-1, self.input_dim)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))

        # h = self.act(self.fc3(h))

        out = self.out(h)
        
        if self.scale is True:
            out = self.tanh_scale * self.scale_act(out)
            # out = self.scale_act(out)
        
        return out

"""
=========================
Classification Networks
=========================
"""

class Discriminator(nn.Module):
    
    def __init__(self, n_layers=5, latent_dim=10, n_units=512, activation='leaky_relu'):
        super(Discriminator, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.n_units = n_units
        
        if activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            self.activation = getattr(F, activation)
        
        self.dense_in = nn.Linear(latent_dim, n_units)
        self.dense1 = nn.Linear(n_units, n_units)
        self.dense2 = nn.Linear(n_units, n_units)
        self.dense3 = nn.Linear(n_units, n_units)
        self.dense4 = nn.Linear(n_units, n_units)
        self.dense5 = nn.Linear(n_units, n_units)

        # theoretically 1 with sigmoid but apparently bad results 
        # => use 2 and softmax
        out_units = 2
        self.dense_out = nn.Linear(n_units, out_units)
        
    
    def forward(self, z):
        
        x = self.activation(self.dense_in(z))
        x = self.activation(self.dense1(x))
        x = self.activation(self.dense2(x))
        x = self.activation(self.dense3(x))
        x = self.activation(self.dense4(x))
        x = self.activation(self.dense5(x))

        out = self.dense_out(x)
        
        return out


class ConvNet(nn.Module):
    def __init__(self, input_dim, activation='relu', n_classes=2):
        """ 
        Convolutional architecture proposed in [1].
        Used to classify shape of DSprites images.
        
        [1] Locatello et. al., "Challenging Common Assumptions
        in the Unsupervised Learning of Disentangled 
        Representations", arXiv:1811.12359 (2018).
        """
        
        super(ConvNet, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes

        im_channels = self.input_dim[0]
        kernel_dim = 4
        hidden_dim = 256
        hidden_channels = 64
        n_ch1 = 32
        n_ch2 = 64
        cnn_kwargs = dict(stride=2, padding=1)
        out_conv_shape = (hidden_channels, kernel_dim, kernel_dim)
        # (leaky_relu, relu, elu)
        self.activation = getattr(F, activation)
        
        self.conv1 = nn.Conv2d(im_channels, n_ch1, kernel_dim, **cnn_kwargs)
        self.conv2 = nn.Conv2d(n_ch1, n_ch1, kernel_dim, **cnn_kwargs)
        self.conv3 = nn.Conv2d(n_ch1, n_ch2, kernel_dim, **cnn_kwargs)
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            self.conv4 = nn.Conv2d(n_ch2, n_ch2, kernel_dim, **cnn_kwargs)
            
        self.dense1 = nn.Linear(np.product(out_conv_shape), hidden_dim)    
        self.dense_out = nn.Linear(hidden_dim, self.n_classes)
        

    def forward(self, x):
        
        # Holds parameters of latent distribution
        # Divided into continuous and discrete dims
        latent_dist = {'continuous': None}

        batch_size = x.size(0)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        
        if self.input_dim[1] == self.input_dim[2] == 64:
            x = self.activation(self.conv4(x))
            
        x = x.view((batch_size, -1))
        x = self.activation(self.dense1(x))
        o = self.dense_out(x)
        
        return o

class SimpleDense(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, n_classes=2):
        super(SimpleDense, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, n_classes)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        h = x.view(-1, self.input_dim)
        h = self.act(self.fc1(h))
        h = self.act(self.fc2(h))
        h = self.act(self.fc3(h))
        
        out = self.out(h)

        return torch.squeeze(out)

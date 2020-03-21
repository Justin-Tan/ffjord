""" Authors: @YannDubs 2019
             @sksq96   2019 """

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from utils import helpers

def compare_histograms_overlay(epoch, itr, data_gen, data_real, save_dir, nbins=50, norm=True, name='plot'):
    # Plot continuum suppression variable distributions for signal, background
    sea_green = '#54ff9f'
    steel_blue = '#4e6bbd'

    sns.distplot(data_gen, color=steel_blue, hist=True, kde=False, norm_hist=norm, label='Generated', bins=nbins,
         hist_kws=dict(edgecolor="0.85", linewidth=0.5, alpha=0.65))

    sns.distplot(data_real, color=sea_green, hist=True, kde=False, norm_hist=norm, label='Real', bins=nbins,
         hist_kws=dict(edgecolor="0.85", linewidth=0.5, alpha=0.8))
        
    plt.autoscale(enable=True, axis='x', tight=False)
    
    if norm:
        plt.ylabel(r'Normalized events/bin')
    else:
        plt.ylabel(r'Events/bin')

    plt.legend(loc="best")
    fig_filename = os.path.join(save_dir, 'figs', '{}_ep_{}_itr_{:04d}.pdf'.format(name, epoch, itr))
    helpers.makedirs(os.path.dirname(fig_filename))
    # plt.show()
    plt.savefig(fig_filename, bbox_inches='tight', format='pdf', dpi=64)
    # plt.savefig('graphs/{}_{}.pdf'.format(name,variable), bbox_inches='tight',format='pdf', dpi=1000)
    plt.gcf().clear()

def visualize_reconstruction(args, data, device, model, epoch, itr):

    model.eval()
        
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)
        recon, latent_sample, latent_dist, flow_output = model(data, sample=True)

        if args.flow == 'no_flow':
            # Sample from distribution defined by amortized encoder parameters - by default
            # diagonal covariance Gaussian
            x_sample = model.reparameterize_continuous(mu=recon['mu'], logvar=recon['logvar']) 
        else:  # using some variant of normalizing flow
            x_flow = flow_output['x_flow']
            if isinstance(x_flow, list):
                x_sample = flow_output['x_flow'][-1]
            else:
                x_sample = x_flow
            
        x = data.cpu().numpy()
        x_sample = x_sample.cpu().numpy()
        for i in range(x_sample.shape[1]):
            try:
                compare_histograms_overlay(epoch=epoch, itr=itr, data_gen=x_sample[:,i],
                    data_real=x[:,i], save_dir=args.snapshot, name='flow_{}'.format(i))
            except ValueError:
                continue

import numpy as np
import pandas as pd
import os, glob, time, datetime
import logging, pickle, argparse

from scipy import stats
from pprint import pprint
from tqdm import tqdm, trange
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Custom modules
import mig_eval
from default_config import args, directories
from models import losses, network, vae
from utils import helpers, initialization, traversals, datasets, evaluate, math, distributions, vis

# FFJORD library
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular, override_divergence_fn


def test(epoch, counter, data, gen_factors, loss_function, device, model, epoch_test_loss, storage, best_test_loss, 
         start_time, epoch_start_time, log_interval_p_epoch, logger):

    model.eval()
        
    with torch.no_grad():
        data = data.to(device, dtype=torch.float)
        gen_factors = gen_factors.to(device)
        
        try:
            recon, latent_sample, latent_dist, flow_output = model(data)
            latent_stats = latent_dist['continuous']
            test_loss = loss_function(data, recon, latent_stats, storage, training=False, latent_sample=latent_sample,
                generative_factors=gen_factors, latent_dist=latent_dist, flow_output=flow_output)
        except ValueError:
            # For losses that require an interior optimization loop
            test_loss = loss_function.call_optimize(data, model, storage, optimizer=None, training=False, 
                                                    generative_factors=gen_factors)
            
        epoch_test_loss += test_loss.item()
        mean_test_loss = epoch_test_loss / counter
        
        best_test_loss = helpers.log(storage, epoch, counter, mean_test_loss, test_loss, 
                                     best_test_loss, start_time, epoch_start_time, 
                                     batch_size=data.shape[0], header='[TEST]', 
                                     log_interval=log_interval_p_epoch, logger=logger)
        
    return best_test_loss, epoch_test_loss


def train(args, model, train_loader, test_loader, device, 
          optimizer, storage, storage_test, logger, log_interval_p_epoch=4):
    
    logger.info('Using device {}'.format(device))
    assert log_interval_p_epoch >= 2, 'Show logs more!'
    log_interval = args.n_data / args.batch_size // log_interval_p_epoch
    assert log_interval > 1, 'Need more time between logs!'
    storage_dir = os.path.join(args.snapshot, 'storage')
    helpers.makedirs(storage_dir)
    
    best_loss, best_test_loss, mean_epoch_loss = np.inf, np.inf, np.inf
    test_loader_iter = iter(test_loader)
    model.train()
    start_time = time.time()
    
    try:
        prior, x_dist = model.prior, model.x_dist
    except AttributeError:
        prior, x_dist = model.module.prior, model.module.x_dist
        
        
    loss_function = losses.get_loss_function(args.loss_type, args=args, log_interval=log_interval, device=device, flow_type=args.flow,
                                             distribution=args.distribution, prior=prior, x_dist=x_dist, supervision=args.supervision,
                                             supervision_lagrange_m=args.supervision_lagrange_m, sensitive_latent_idx=args.sensitive_latent_idx)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, 
                                                           factor=0.5, verbose=True)
    
    for epoch in trange(args.n_epochs, desc='Epoch'):

        epoch_loss = []
        epoch_test_loss = 0.
        counter = 0
        epoch_start_time = time.time()
        
        if epoch % args.save_interval == 0 and epoch > 1 and epoch != args.n_epochs:
            helpers.save_model(model, optimizer, mean_epoch_loss, args.checkpoints_save, epoch, device, args=args)
        
        for idx, (data, gen_factors) in enumerate(tqdm(train_loader, desc='Train'), 0):

            data = data.to(device, dtype=torch.float)
            gen_factors = gen_factors.to(device)
            
            try:
                recon, latent_sample, latent_dist, flow_output = model(data)
                latent_stats = latent_dist['continuous']
                

                loss = loss_function(data, recon, latent_stats, storage=storage, training=True, latent_sample=latent_sample,
                    generative_factors=gen_factors, latent_dist=latent_dist, flow_output=flow_output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            except ValueError:
                # For losses that require an interior optimization loop
                loss = loss_function.call_optimize(data, model, storage, optimizer, training=model.training,
                                                   generative_factors=gen_factors)

            epoch_loss.append(loss.item())

            if idx % log_interval == 1 and idx > 1:

                counter += 1
                mean_epoch_loss = np.mean(epoch_loss)
                best_loss = helpers.log(storage, epoch, counter, mean_epoch_loss, loss.item(),
                                best_loss, start_time, epoch_start_time, batch_size=data.shape[0],
                                log_interval=log_interval_p_epoch)
                try:
                    test_data, test_gen_factors = test_loader_iter.next()
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    test_data, test_gen_factors = test_loader_iter.next()

                # temporary check of logvar
                z_logvar = latent_stats[-1]
                x_logvar = recon['logvar']
                logger.info('Z_LOGVAR avg. {}'.format(z_logvar.mean().item()))
                logger.info('X_LOGVAR avg. {}'.format(x_logvar.mean().item()))
                best_test_loss, epoch_test_loss = test(epoch, counter, test_data, test_gen_factors,
                                                       loss_function, device, model, epoch_test_loss, 
                                                       storage_test, best_test_loss, start_time, epoch_start_time,
                                                       log_interval_p_epoch, logger)

                if args.smoke_test:
                   return None 

                with open(os.path.join(storage_dir, 'storage_{}_tmp.pkl'.format(args.name)), 'wb') as handle:
                    pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Visualization
                vis.visualize_reconstruction(args, data, device, model, epoch, idx)

                model.train()

        mean_epoch_loss = np.mean(epoch_loss)
        mean_test_loss = epoch_test_loss / counter
        scheduler.step(mean_test_loss)
        logger.info('===>> Epoch {} | Mean train loss: {:.3f} | Mean test loss: {:.3f}'.format(epoch, 
            mean_epoch_loss, mean_test_loss))    
    
    with open(os.path.join(storage_dir, 'storage_{}_{:%Y_%m_%d_%H:%M:%S}.pkl'.format(args.name, datetime.datetime.now())), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    ckpt_path = helpers.save_model(model, optimizer, mean_epoch_loss, args.checkpoints_save, epoch, device, args=args)
    logger.info("Training complete. Mean time / epoch: {:.3f}".format((time.time()-start_time)/args.n_epochs))
    
    return ckpt_path


if __name__ == '__main__':

    description = "Learning invariant representations with disentangling variational autoencoders, implemented in Pytorch."
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General options
    general = parser.add_argument_group('General options')
    general.add_argument("-n", "--name", default=args.name, help="Identifier for checkpoints and metrics.")
    general.add_argument("-o", "--output_directory", default=directories.checkpoints, help="Directory to store checkpoints and logs.")
    general.add_argument("-d", "--dataset", type=str, default='dsprites', help="Training dataset to use.",
        choices=['dsprites', 'custom', 'dsprites_scream'], required=True)
    general.add_argument("-gpu", "--gpu", type=int, default=0, help="GPU ID.")
    general.add_argument("-lpe", "--logs_per_epoch", type=int, default=4, help="Number of times to report metrics per epoch.")
    general.add_argument("-save_intv", "--save_interval", type=int, default=8, help="Number of epochs between checkpointing.")
    general.add_argument("-multigpu", "--multigpu", help="Toggle data parallel capability using torch DataParallel", action="store_true")
    general.add_argument('-bs', '--batch_size', type=int, default=2048, help='input batch size for training')
    general.add_argument('--save', type=str, default='experiments', help='Parent directory for stored information')
    general.add_argument('--smoke_test', action='store_true', help='Shut up and train! No extra metrics.')
    general.add_argument(
        '-f', '--flow', type=str, default='no_flow', choices=['cnf', 'cnf_amort', 'real_nvp', 'no_flow'], 
        help="""Type of flows to use in decoder of VAE, no flows can also be selected."""
    )

    # Optimization-related options
    optim = parser.add_argument_group("Optimization-related options")
    optim.add_argument('-epochs', '--n_epochs', type=int, default=32, help="Number of passes over training dataset.")
    optim.add_argument("-lr", "--learning_rate", type=float, default=args.learning_rate, help="Optimizer learning rate.")
    optim.add_argument("-wd", "--weight_decay", type=float, default=args.weight_decay, help="Coefficient of L2 regularization.")

    # Model options
    model = parser.add_argument_group("Model-related options")
    model.add_argument("-l", "--loss_type", type=str, default=args.loss_type, help="Choice of loss function.",
        choices=['VAE', 'beta_VAE', 'annealed_VAE', 'factor_VAE', 'beta_TCVAE', 'beta_TCVAE_sensitive'], required=True)
    model.add_argument("-z", "--latent_dim", type=int, default=args.latent_dim, help="Dimension of latent space.")
    model.add_argument("-sv", "--supervision", help="Apply supervision to VAE loss.", action="store_true")
    model.add_argument("-lambda", "--supervision_lagrange_m", type=int, default=args.supervision_lagrange_m,
                       help="Lagrange multiplier for supervised component of loss.")
    model.add_argument("-s_idx", "--sensitive_latent_idx", type=int, nargs='+', default=args.sensitive_latent_idx,
                       help="Indices of latent dimensions corresponding to sensitive factors.")

    # Beta-vae options
    beta_vae = parser.add_argument_group("Beta-VAE-related options")
    beta_vae.add_argument("-beta", "--beta", type=float, default=args.beta, help="Coefficient of KL term in ELBO for Beta-VAE.")

    # Annealed-vae options
    annealed_vae = parser.add_argument_group("Annealed-VAE-related options")
    annealed_vae.add_argument("-gamma", "--gamma", type=float, default=args.gamma, help="Coefficient of annealed KL term for Annealed-VAE.")

    # Factor-vae options
    fvae = parser.add_argument_group("Factor-VAE-related options")
    fvae.add_argument("-gamma_fvae", "--gamma_fvae", type=float, default=args.gamma_fvae, help="Coefficient of TC estimate for Factor-VAE.")

    # beta-tcvae / beta-tcvae-sensitive options
    btcvae = parser.add_argument_group("BTC-VAE / BTC-VAE-Sensitive - related options")
    btcvae.add_argument("-alpha_btcvae", "--alpha_btcvae", type=float, default=args.alpha_btcvae, help="Alpha coefficient in beta-TCVAE loss.")
    btcvae.add_argument("-beta_btcvae", "--beta_btcvae", type=float, default=args.beta_btcvae, help="Beta coefficient in beta-TCVAE loss.")
    btcvae.add_argument("-gamma_btcvae", "--gamma_btcvae", type=float, default=args.gamma_btcvae, help="Gamma coefficient in beta-TCVAE loss.")

    # Continuous-time normalizing flow options
    cnf_args = parser.add_argument_group("CNF - related options")
    cnf_args.add_argument('--dims', type=str, default='256-256', help='Defines dynamics network architecture')
    cnf_args.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
    cnf_args.add_argument('--time_length', type=float, default=0.5)
    cnf_args.add_argument('--train_T', type=eval, default=False)
    cnf_args.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
    cnf_args.add_argument("--nonlinearity", type=str, default="softplus", choices=odefunc.NONLINEARITIES)
    cnf_args.add_argument('-r', '--rank', type=int, default=1)

    SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
    cnf_args.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
    cnf_args.add_argument('--atol', type=float, default=1e-5)
    cnf_args.add_argument('--rtol', type=float, default=1e-5)
    cnf_args.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")
    cnf_args.add_argument(
        "--layer_type", type=str, default="concat",
        choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
    )

    cnf_args.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
    cnf_args.add_argument('--test_atol', type=float, default=None)
    cnf_args.add_argument('--test_rtol', type=float, default=None)

    cnf_args.add_argument('--residual', type=eval, default=False, choices=[True, False])
    cnf_args.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
    cnf_args.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
    cnf_args.add_argument('--bn_lag', type=float, default=0)

    cnf_args.add_argument('--evaluate', action='store_true')
    cnf_args.add_argument('--model_path', type=str, default='')

    cmd_args = parser.parse_args()

    if cmd_args.gpu != 0:
        torch.cuda.set_device(cmd_args.gpu)

    start_time = time.time()
    device = helpers.get_device()

    # Override default arguments with provided command line arguments
    dictify = lambda x: dict((n, getattr(x, n)) for n in dir(x) if not (n.startswith('__') or 'logger' in n))
    args_d, cmd_args_d = dictify(args), vars(cmd_args)
    args_d.update(cmd_args_d)
    args = helpers.Struct(**args_d)
    args.latent_spec['continuous'] = cmd_args.latent_dim
    if args.output_directory not in [directories.checkpoints, 'results']:
        args.output_directory = os.path.join('checkpoints', args.output_directory)

    args.name = '{}_{}'.format(args.loss_type, args.dataset)
    args = helpers.setup_signature(args)
    logger = helpers.logger_setup(logpath=os.path.join(args.snapshot, 'logs'), filepath=os.path.abspath(__file__))
    logger.info('SAVING LOGS/CHECKPOINTS/RECORDS TO {}'.format(args.snapshot))
    logger.info('Using GPU ID {}'.format(args.gpu))

    assert args.loss_type in args.LOSSES, 'Unrecognized loss type!'
    assert args.dataset in args.DATASETS, 'Unrecognized dataset!'
    logger.info('Using dataset: {}'.format(args.dataset))
    test_loader = datasets.get_dataloaders(args.dataset,
                                batch_size=args.batch_size,
                                logger=logger,
                                train=False,
                                sampling_bias=args.sampling_bias,
                                shuffle=args.shuffle)

    all_loader = datasets.get_dataloaders(args.dataset,
                                batch_size=args.batch_size,
                                logger=logger,
                                metrics=True,
                                sampling_bias=args.sampling_bias,
                                shuffle=args.shuffle)


    if args.supervision is False:
        logger.info('Supervision is OFF')
        train_loader = all_loader
    else:
        logger.info('Supervision is ON')
        train_loader = datasets.get_dataloaders(args.dataset,
                                    batch_size=args.batch_size,
                                    logger=logger,
                                    train=True,
                                    sampling_bias=args.sampling_bias,
                                    shuffle=args.shuffle)

        K = all_loader.dataset.n_gen_factors
        if len(args.sensitive_latent_idx) > K:
            args.sensitive_latent_idx = args.sensitive_latent_idx[:K]
        logger.info('Applying supervision to the following latent indices: {}'.format(args.sensitive_latent_idx))

    args.n_data = len(train_loader.dataset)

    try:
        args.input_dim = datasets.get_img_size(args.dataset)
    except AttributeError:
        args.input_dim = all_loader.dataset.input_dim
    logger.info('Input Dimensions: {}'.format(args.input_dim))

    if args.dataset == 'custom':
        args.x_dist = 'normal'

    if args.flow == 'no_flow':
        model = vae.VAE(args)
    elif args.flow == 'real_nvp':
        model = vae.realNVP_VAE(args)
    elif args.flow == 'cnf':
        model = vae.VAE_ODE(args)
    elif args.flow == 'cnf_amort':
        model = vae.VAE_ODE_amortized(args)

    logger.info(model)

    n_gpus = torch.cuda.device_count()
    if 'cnf' not in args.flow:
        helpers.summary(model, input_size=[[args.input_dim]] if args.dataset=='custom' else args.input_dim, device='cpu')

    if n_gpus > 1 and args.multigpu is True:
        logger.info('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    try:
        args.latent_dim = model.latent_dim
    except AttributeError:
        args.latent_dim = model.module.latent_dim

    metadata = dict(input_dim=args.input_dim, latent_dim=args.latent_dim,
                    model_loss=args.loss_type)

    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    metadata.update(args_d)
    logger.info(metadata)


    """
    Train
    """
    storage = defaultdict(list)
    storage_test = defaultdict(list)
    ckpt_path = train(args, model, train_loader, test_loader, device, optimizer, storage, storage_test,
        logger, log_interval_p_epoch=args.logs_per_epoch)
    args.ckpt = ckpt_path

    """
    Generate metrics
    """

    if args.dataset == 'custom':
        metrics, df_mbc, df = mig_eval.metric_custom(args, model, device, logger, train_loader=train_loader,
            test_loader=test_loader, storage=storage_test, evaluate_with_mbc=True)
    else:
        metric, discrete_metric, marginal_entropies, conditional_entropies, z_v_mapping = mig_eval.mutual_info_metric_shapes(args,
            model, torch.device('cuda:1'), logger, storage=storage_test, gpu_id=1)

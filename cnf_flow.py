import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import os
import time
import pickle
import argparse
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.datasets

from collections import defaultdict
from tqdm import tqdm, trange, tqdm_notebook

# ffjord lib
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform, compare_histograms_overlay
import lib.layers.odefunc as odefunc

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular, override_divergence_fn

from diagnostics.viz_toy import save_trajectory, trajectory_to_video

# vae lib
from models import losses, network, vae
from utils import helpers, datasets, math, distributions

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--dataset', choices=['custom'], type=str, default='custom'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='256-256-256')
parser.add_argument('--hdim_factor', type=int, default=10, help='Multiplying factor between data, hidden dim.')
parser.add_argument('--nhidden', type=int, default=1, help='Number of hidden layers defining network dynamics.')
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs (flow steps).')
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="approximate", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="softplus", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-8)
parser.add_argument('--rtol', type=float, default=1e-6)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=True, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--early_stopping', type=int, default=32)
parser.add_argument('--n_epochs', type=int, default=32)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--test_batch_size', type=int, default=2048)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=1e-5)

# Track quantities
parser.add_argument('--l1int', type=float, default=None, help="int_t ||f||_1")
parser.add_argument('--l2int', type=float, default=None, help="int_t ||f||_2")
parser.add_argument('--dl2int', type=float, default=None, help="int_t ||f^T df/dt||_2")
parser.add_argument('--JFrobint', type=float, default=None, help="int_t ||df/dx||_F")
parser.add_argument('--JdiagFrobint', type=float, default=None, help="int_t ||df_i/dx_i||_F")
parser.add_argument('--JoffdiagFrobint', type=float, default=None, help="int_t ||df/dx - df_i/dx_i||_F")

parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--save', type=str, default='experiments/cnf')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--viz_freq', type=int, default=1000)
parser.add_argument('--val_freq', type=int, default=250)
parser.add_argument('--log_freq', type=int, default=50)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--multigpu', action='store_true')

args = parser.parse_args()
assert args.viz_freq > args.val_freq and args.viz_freq % args.val_freq == 0
# logger
utils.makedirs(args.save)
storage_dir = os.path.join(args.save, 'storage')
utils.makedirs(storage_dir)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0
    args.train_T = False

logger.info(args)
ndecs = 0

def get_transforms(model):

    def sample_fn(z, logpz=None):
        if logpz is not None:
            return model(z, logpz, reverse=True)
        else:
            return model(z, reverse=True)

    def density_fn(x, logpx=None):
        if logpx is not None:
            return model(x, logpx, reverse=False)
        else:
            return model(x, reverse=False)

    return sample_fn, density_fn

def update_lr(optimizer, n_vals_without_improvement, logger):
    global ndecs
    print('Cycles without improvement:', n_vals_without_improvement)
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping // 3:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 4
        ndecs = 1
        logger.info('REDUCING LEARNING RATE TO {}'.format(args.lr/4))
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping // 3 * 2:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / 16
        ndecs = 2
        logger.info('REDUCING LEARNING RATE TO {}'.format(args.lr/16))
    else:
        if ndecs > 0:
            for param_group in optimizer.param_groups:
                param_group["lr"] = args.lr / 4**ndecs
            logger.info('REDUCING LEARNING RATE TO {}'.format(args.lr/(4**ndecs)))


def get_data(args, logger):
    test_loader = datasets.get_dataloaders(args.dataset,
                               batch_size=args.batch_size,
                               logger=logger,
                               train=False,
                               shuffle=False)


    train_loader = datasets.get_dataloaders(args.dataset,
                                batch_size=args.batch_size,
                                logger=logger,
                                train=True,
                                shuffle=True)
    args.n_data = len(train_loader.dataset)

    return train_loader, test_loader


def get_regularization_loss(model, regularization_fns, regularization_coeffs):
    if len(regularization_coeffs) > 0:
        reg_states = get_regularization(model, regularization_coeffs)
        reg_loss = sum(
            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
        )

    return reg_loss, reg_states

def compute_loss(x, model, batch_size=None):
    if batch_size is None: batch_size = args.batch_size

    # load data
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z by running model forward
    z, delta_logp = model(x, zero)

    # compute log q(z)
    # logpz = standard_normal_logprob(z).sum(1, keepdim=True)
    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss

def train_ffjord(model, optimizer, device, logger, iterations=8000):
    print('Using device', device)

    end = time.time()
    best_loss = float('inf')
    val_itr = 0
    n_vals_without_improvement = 0
    model.train()
    storage = defaultdict(list)


    for epoch in trange(args.n_epochs, desc='epoch'):

        epoch_loss = []
        epoch_test_loss = 0.
        counter = 0
        epoch_start_time = time.time()

        if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
            if epoch >= 2:
                break

        for itr, (data, gen_factors) in enumerate(tqdm(train_loader, desc='Train'), 0):
            if args.early_stopping > 0 and n_vals_without_improvement > args.early_stopping:
                break

            x = cvt(data)

            optimizer.zero_grad()
            if args.spectral_norm: spectral_norm_power_iteration(model, 1)

            loss = compute_loss(x, model)
            loss_meter.update(loss.item())

            if len(regularization_coeffs) > 0:
                reg_loss, reg_states = get_regularization_loss(model, regularization_fns, regularization_coeffs)
                loss = loss + reg_loss

            total_time = count_total_time(model)
            nfe_forward = count_nfe(model)

            loss.backward()
            optimizer.step()

            nfe_total = count_nfe(model)
            nfe_backward = nfe_total - nfe_forward
            nfef_meter.update(nfe_forward)
            nfeb_meter.update(nfe_backward)
            time_meter.update(time.time() - end)
            tt_meter.update(total_time)

            end = time.time()

            if itr % args.log_freq == 0:
                storage['time'].append(time.time())
                storage['train_loss'].append(loss_meter.avg)
                log_message = (
                    'Epoch {} | Iter {} | Time {:.3f}({:.3f}) | Loss {:.3f}({:.3f}) | NFE Forward {:.0f}({:.1f})'
                    ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.3f}({:.3f})'.format(
                        epoch, itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                        nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
                    )
                )

                if len(regularization_coeffs) > 0:
                    log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)

                logger.info(log_message)

            end = time.time()


            if itr % args.val_freq == 0:
                improved = '[]'
                model.eval()
                start_time = time.time()
                val_x = list()
                with torch.no_grad():
                    val_loss_meter = utils.AverageMeter()
                    val_nfe_meter = utils.AverageMeter()
                    
                    sample_fn, density_fn = get_transforms(model)

                    for (x, gen_factors) in tqdm(itertools.islice(test_loader, 10*val_itr, 10*(val_itr+1)), desc='val'):
                        x = cvt(x)
                        val_x.append(x)
                        val_loss = compute_loss(x, model)
                        val_nfe = count_nfe(model)
                        val_loss_meter.update(val_loss.item(), x.shape[0])
                        val_nfe_meter.update(val_nfe)

                    # Visualization
                    if (itr % args.viz_freq == 0) and (itr > 500):
                        val_x = torch.cat(val_x, axis=0).cpu().numpy()
                        val_z = cvt(torch.randn(val_x.shape))

                        # Transform base distribution to x by running model backward
                        val_sample = sample_fn(val_z)
                        val_sample = val_sample.cpu().numpy()
                        for i in range(val_sample.shape[1]):
                            compare_histograms_overlay(epoch=epoch, itr=itr, data_gen=val_sample[:,i],
                                data_real=val_x[:,i], save_dir=args.save, name='cnf_{}'.format(i))

                    if val_loss_meter.avg < best_loss:
                        best_loss = val_loss_meter.avg
                        improved = '[*]'
                        utils.makedirs(args.save)
                        torch.save({
                            'args': args,
                            'state_dict': model.state_dict(),
                        }, os.path.join(args.save, 'cnf_hep_ckpt.pth'))
                        n_vals_without_improvement = 0
                    else:
                        n_vals_without_improvement += 1
                    update_lr(optimizer, n_vals_without_improvement, logger)

                    storage['val_loss'].append(val_loss_meter.avg)
                    log_message = (
                        '[VAL] Epoch {} | Val Loss {:.3f} | NFE {:.0f} | '
                        'NoImproveEpochs {:02d}/{:02d} {}'.format(
                            epoch, val_loss_meter.avg, val_nfe_meter.avg, n_vals_without_improvement, 
                            args.early_stopping, improved
                        )
                    )
                    logger.info(log_message)
                    val_itr += 1

                    if (val_itr+1)*10 > len(test_loader):
                        val_itr = 0

                model.train()

                with open(os.path.join(storage_dir, 'storage_tmp.pkl'), 'wb') as handle:
                    pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)


    logger.info('Training has finished.')
    model = helpers.quick_restore_model(model, os.path.join(args.save, 'cnf_hep_ckpt.pth')).to(device)
    set_cnf_options(args, model)

    time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
    with open(os.path.join(storage_dir, 'storage_end_{}.pkl'.format(time_signature)), 'wb') as handle:
        pickle.dump(storage, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('Evaluating model on test set.')
    model.eval()

    override_divergence_fn(model, "brute_force")

    with torch.no_grad():
        test_loss = utils.AverageMeter()
        test_nfe = utils.AverageMeter()

        for itr, (x, gen_factors) in enumerate(tqdm(test_loader, desc='Test'), 0): 
            x = cvt(x)
            test_loss.update(compute_loss(x, model).item(), x.shape[0])
            test_nfe.update(count_nfe(model))

    log_message = '[TEST] Iter {} | Test Loss {:.3f} | NFE {:.0f}'.format(itr, test_loss.avg, test_nfe.avg)
    logger.info(log_message)

if __name__ == '__main__':

    if args.gpu != 0:
        torch.cuda.set_device(args.gpu)

    # device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    device = helpers.get_device()

    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)
    regularization_fns, regularization_coeffs = create_regularization_fns(args)

    train_loader, test_loader = get_data(args, logger)
    input_dim = train_loader.dataset.input_dim
    # args.dims = '-'.join([str(args.hdim_factor * input_dim)] * args.nhidden)
    args.dims = '256-256-256'
    # args.dims = '512-512-512'

    model = build_model_tabular(args, input_dim, regularization_fns).to(device)
    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(args, model)

    for k in model.state_dict().keys():
        logger.info(k)

    logger.info(model)
    n_gpus = torch.cuda.device_count()
    logger.info('Using {} GPUs.'.format(n_gpus))
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))

    if n_gpus > 1 and args.multigpu is True:
        print('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.96)
    loss_meter = utils.RunningAverageMeter(0.96)
    nfef_meter = utils.RunningAverageMeter(0.96)
    nfeb_meter = utils.RunningAverageMeter(0.96)
    tt_meter = utils.RunningAverageMeter(0.96)

    train_ffjord(model, optimizer, device, logger)

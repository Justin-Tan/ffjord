import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.datasets

from tqdm import tqdm, trange, tqdm_notebook

# ffjord lib
import lib.toy_data as toy_data
import lib.utils as utils
from lib.visualize_flow import visualize_transform
import lib.layers.odefunc as odefunc

from train_misc import standard_normal_logprob
from train_misc import set_cnf_options, count_nfe, count_parameters, count_total_time
from train_misc import add_spectral_norm, spectral_norm_power_iteration
from train_misc import create_regularization_fns, get_regularization, append_regularization_to_log
from train_misc import build_model_tabular, override_divergence_fn

from diagnostics.viz_toy import save_trajectory, trajectory_to_video

# vae lib
from utils import helpers, datasets, math, distributions

SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams', 'fixed_adams']
parser = argparse.ArgumentParser('Continuous Normalizing Flow')
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings', 'hep'],
    type=str, default='pinwheel'
)
parser.add_argument(
    "--layer_type", type=str, default="concatsquash",
    choices=["ignore", "concat", "concat_v2", "squash", "concatsquash", "concatcoord", "hyper", "blend"]
)
parser.add_argument('--dims', type=str, default='64-64-64')
parser.add_argument('--hdim_factor', type=int, default=10)
parser.add_argument('--nhidden', type=int, default=1)
parser.add_argument("--num_blocks", type=int, default=1, help='Number of stacked CNFs.')
parser.add_argument('--time_length', type=float, default=1.0)
parser.add_argument('--train_T', type=eval, default=True)
parser.add_argument("--divergence_fn", type=str, default="brute_force", choices=["brute_force", "approximate"])
parser.add_argument("--nonlinearity", type=str, default="tanh", choices=odefunc.NONLINEARITIES)

parser.add_argument('--solver', type=str, default='dopri5', choices=SOLVERS)
parser.add_argument('--atol', type=float, default=1e-5)
parser.add_argument('--rtol', type=float, default=1e-5)
parser.add_argument("--step_size", type=float, default=None, help="Optional fixed step size.")

parser.add_argument('--test_solver', type=str, default=None, choices=SOLVERS + [None])
parser.add_argument('--test_atol', type=float, default=None)
parser.add_argument('--test_rtol', type=float, default=None)

parser.add_argument('--residual', type=eval, default=False, choices=[True, False])
parser.add_argument('--rademacher', type=eval, default=False, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batch_norm', type=eval, default=False, choices=[True, False])
parser.add_argument('--bn_lag', type=float, default=0)

parser.add_argument('--early_stopping', type=int, default=16)
parser.add_argument('--niters', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=1024)
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
parser.add_argument('--viz_freq', type=int, default=100)
parser.add_argument('--val_freq', type=int, default=200)
parser.add_argument('--log_freq', type=int, default=10)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# logger
utils.makedirs(args.save)
logger = utils.get_logger(logpath=os.path.join(args.save, 'logs'), filepath=os.path.abspath(__file__))

if args.layer_type == "blend":
    logger.info("!! Setting time_length from None to 1.0 due to use of Blend layers.")
    args.time_length = 1.0
    args.train_T = False

logger.info(args)

def compute_loss(args, model, batch_size=None):
    if batch_size is None: batch_size = args.batch_size

    # load data
    # x = toy_data.inf_train_gen(args.data, batch_size=batch_size)
    x = sklearn.datasets.make_moons(n_samples=256, noise=.05)[0].astype(np.float32)
    x = torch.from_numpy(x).type(torch.float32).to(device)
    zero = torch.zeros(x.shape[0], 1).to(x)

    # transform to z
    z, delta_logp = model(x, zero)

    # compute log q(z)
    # logpz = standard_normal_logprob(z).sum(1, keepdim=True)
    logpz = standard_normal_logprob(z).view(z.shape[0], -1).sum(1, keepdim=True)  # logp(z)
    logpx = logpz - delta_logp
    loss = -torch.mean(logpx)
    return loss

def get_regularization_loss(model, regularization_fns, regularization_coeffs):
    if len(regularization_coeffs) > 0:
        reg_states = get_regularization(model, regularization_coeffs)
        reg_loss = sum(
            reg_state * coeff for reg_state, coeff in zip(reg_states, regularization_coeffs) if coeff != 0
        )

    return reg_loss

def train_moons_ffjord(model, optimizer, device, logger, iterations=8000):
    print('Using device', device)

    end = time.time()
    best_loss = float('inf')
    model.train()

    for itr in trange(iterations, desc='Itr'):
        optimizer.zero_grad()
        if args.spectral_norm: spectral_norm_power_iteration(model, 1)

        loss = compute_loss(args, model)
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            reg_loss = get_regularization_loss(model, regularization_fns, regularization_coeffs)
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

        log_message = (
            'Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | NFE Forward {:.0f}({:.1f})'
            ' | NFE Backward {:.0f}({:.1f}) | CNF Time {:.4f}({:.4f})'.format(
                itr, time_meter.val, time_meter.avg, loss_meter.val, loss_meter.avg, nfef_meter.val, nfef_meter.avg,
                nfeb_meter.val, nfeb_meter.avg, tt_meter.val, tt_meter.avg
            )
        )

        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)


        if itr % args.val_freq == 0 or itr == args.niters:
            logger.info(log_message)
            improved = '[]'

            with torch.no_grad():
                model.eval()
                test_loss = compute_loss(args, model, batch_size=args.test_batch_size)
                test_nfe = count_nfe(model)

                if test_loss.item() < best_loss:
                    best_loss = test_loss.item()
                    improved = '[*]'

                log_message = '[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f} {}'.format(itr, test_loss, test_nfe, improved)
                logger.info(log_message)

            model.train()
        
        end = time.time()

    logger.info('Training has finished.')


if __name__ == '__main__':

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    cvt = lambda x: x.type(torch.float32).to(device, non_blocking=True)
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, 2, regularization_fns).to(device)
    if args.spectral_norm: add_spectral_norm(model)
    set_cnf_options(args, model)

    logger.info(model)
    n_gpus = torch.cuda.device_count()
    logger.info('Using {} GPUs.'.format(n_gpus))
    logger.info("Number of trainable parameters: {}".format(count_parameters(model)))
    if n_gpus > 1:
        print('Using {} GPUs.'.format(n_gpus))
        model = nn.DataParallel(model)
        args.multigpu = True
    else:
        args.multigpu = False
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = utils.RunningAverageMeter(0.93)
    loss_meter = utils.RunningAverageMeter(0.93)
    nfef_meter = utils.RunningAverageMeter(0.93)
    nfeb_meter = utils.RunningAverageMeter(0.93)
    tt_meter = utils.RunningAverageMeter(0.93)

    train_moons_ffjord(model, optimizer, device, logger, iterations=8000)
import os
import time
import argparse
import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm, trange, tqdm_notebook
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, roc_auc_score, mutual_info_score

# My imports
from models import network, vae
from utils import helpers, datasets, math, distributions

def qzCx_sample(params):
    """
    Sample from posterior q(z|x).
    Assume normal with diagonal covariance.  
    Params: [B, LD, 2]
    Expect mu = params[..., 0]
           logvar = params[..., 1]
    """

    mu = params.select(-1, 0)
    logvar = params.select(-1, 1)
    sigma_sqrt = torch.exp(0.5 * logvar)
    epsilon = torch.randn_like(sigma_sqrt)
    return mu + sigma_sqrt * epsilon


def estimate_entropies(qzCx_samples, qzCx_params, device, n_samples=10000, weights=None):
    """Computes the term:
        E_{p(x)} E_{q(z|x)} [-log q(z)]
    and
        E_{p(x)} E_{q(z_j|x)} [-log q(z_j)]
    where marginal q(z) = 1/N sum_n=1^N q(z|x_n).

    Assumes samples are from q(z|x) for *all* x in the dataset.
    Assumes that q(z|x) is factorial ie. q(z|x) = prod_j q(z_j|x).

    Computes numerically stable NLL:
        - log q(z) = log N - logsumexp_n=1^N log q(z|x_n)

    Inputs:
    -------
        qzCx_samples (K, N) Variable
        qzCx_params  (N, K, nparams) Variable
        weights (N) Variable

        Where:
        N: Dataset size
        K: Number of generative factors
    """

    qzCx_samples, qzCx_params = qzCx_samples.to(device), qzCx_params.to(device)
    # Only take a sample subset of the samples
    if weights is None:
        qzCx_samples = qzCx_samples.index_select(1, torch.randperm(qzCx_samples.size(1))[:n_samples].to(device))
    else:
        sample_inds = torch.multinomial(weights, n_samples, replacement=True)
        qzCx_samples = qzCx_samples.index_select(1, sample_inds)

    K, S = qzCx_samples.size()
    N, _, nparams = qzCx_params.size()
    assert(K == qzCx_params.size(1))

    log_N = np.log(N)
    if weights is None:
        weights = - log_N
    else:
        weights = torch.log(weights.view(N, 1, 1) / weights.sum())
    
    entropies = torch.zeros(K).to(device)
    pbar = tqdm(total=S)
    k = 0
    
    # Low max. batch size otherwise memory issues
    batch_size = 64
    while k < S:
        batch_size = min(batch_size, S - k)

        params = qzCx_params.view(N, K, 1, nparams).expand(N, K, S, nparams)[:, :, k:k + batch_size]
        mu, logvar = params.select(-1,0), params.select(-1,1)
        log_qzCx_i = math.log_density_gaussian(
            qzCx_samples.view(1, K, S).expand(N, K, S)[:, :, k:k + batch_size], mu=mu, logvar=logvar)
        k += batch_size

        # computes - log q(z_i) summed over minibatch
        # H(z) = - E_{q(z|x)p(x)}[1/N * \sum_x log q(z|x)]
        
        entropies += - torch.logsumexp(log_qzCx_i + weights, dim=0, keepdim=False).data.sum(1)
#         log_qz = - log_N + torch.logsumexp(log_qzCx_i + weights, dim=0, keepdim=False).data
#         entropies += - log_qz.sum(1)
        
        pbar.update(batch_size)
        
    pbar.close()

    entropies /= S

    return entropies

def compute_metric_shapes(marginal_entropies, cond_entropies):
    """
    z, v: Learned latent and true generative factors, respectively.
    Marginal entropies: H(z)
    Conditional entropies: H(z|v) --> I(z;v) = H(z) - H(z|v)
    """
    factor_entropies = [6, 40, 32, 32]
    mutual_infos = torch.unsqueeze(marginal_entropies, dim=0) - cond_entropies
    mutual_infos = torch.sort(mutual_infos, dim=1, descending=True)[0].clamp(min=0)
    mi_normed = mutual_infos / torch.Tensor(factor_entropies).log()[:, None]
    
    MIG = torch.mean(mi_normed[:,0] - mi_normed[:,1])
    return MIG, mutual_infos


def _histogram_discretize(target, num_bins=20):
    """Discretization based on histograms."""
    discretized = np.zeros_like(target)
    for i in range(target.shape[0]):
        discretized[i, :] = np.digitize(target[i, :], np.histogram(
            target[i, :], num_bins)[1][:-1])
    return discretized

def discrete_mutual_info(z, v):
    """
    Compute discrete mutual information.
    z: array-like
        Matrix of learned latent codes, shape [LD, B] where LD is the 
        latent dimension. z is taken to be the mean value of the latent
        representation.
    """
    num_codes = z.shape[0]
    num_factors = v.shape[0]
    mi = np.zeros([num_codes, num_factors])
    for i in range(num_codes):
        for j in range(num_factors):
            mi[i, j] = mutual_info_score(v[j, :], z[i, :])
    return mi

def discrete_entropy(v):
    """
    Compute discrete mutual information.
    v: array-like
        Matrix of underlying generative factors, shape [N,B]
        for N ground truth factors
    """
    num_factors = v.shape[0]
    H = np.zeros(num_factors)
    for j in range(num_factors):
        H[j] = mutual_info_score(v[j, :], v[j, :])
    return H


def estimate_MIG_discrete(qzCx_samples, gen_factors):
    
    z_discrete = _histogram_discretize(qzCx_samples, num_bins=20)
    mi_discrete = discrete_mutual_info(np.transpose(z_discrete), np.transpose(gen_factors))
    H_v = discrete_entropy(np.transpose(gen_factors))
    sorted_mi_discrete = np.sort(mi_discrete, axis=0)[::-1]
    discrete_metric = np.mean(np.divide(sorted_mi_discrete[0, :] - sorted_mi_discrete[1, :],H_v[:]))
    
    top_mi_normed_discrete = np.divide(sorted_mi_discrete[0,:], H_v)
    top_mi_idx_discrete = np.argsort(mi_discrete, axis=0)[::-1][0,:]
    
    return discrete_metric, top_mi_normed_discrete, top_mi_idx_discrete

def estimate_MIG_discrete_custom(qzCx_samples, gen_factors, nbins=24, n_samples=10000, multidim=False):
    
    ridx = torch.randperm(qzCx_samples.size(0))[:n_samples]
    qzCx_samples = qzCx_samples.index_select(dim=0, index=ridx)
    gen_factors = gen_factors.index_select(dim=0, index=ridx).numpy()
    
    minmaxscaler = MinMaxScaler()
    gen_factors = minmaxscaler.fit_transform(gen_factors)
    
    z_discrete = _histogram_discretize(qzCx_samples, num_bins=nbins)
    
    if multidim is False:
        # Beam-constrained mass only
        gen_factors = gen_factors[:,0]
        gen_factors = np.expand_dims(np.digitize(gen_factors, bins=np.linspace(0.,1.,nbins), right=False), 1)
    else:
        gen_factors = np.stack([np.digitize(gen_factors[:,i], bins=np.linspace(0.,1.,nbins), right=False) for i in range(gen_factors.shape[-1])], axis=1)

    mi_discrete = discrete_mutual_info(np.transpose(z_discrete), np.transpose(gen_factors))
    H_v = discrete_entropy(np.transpose(gen_factors))
    sorted_mi_discrete = np.sort(mi_discrete, axis=0)[::-1]
    discrete_metric = np.mean(np.divide(sorted_mi_discrete[0, :] - sorted_mi_discrete[1, :],H_v[:]))
    top_mi_normed_discrete = np.divide(sorted_mi_discrete[0,:], H_v), 
    top_mi_idx_discrete = np.argsort(mi_discrete, axis=0)[::-1][0,:]
    
    return discrete_metric, top_mi_normed_discrete, top_mi_idx_discrete


def report(args, logits, labels, loss_scalar, start, logger, header='[TRAIN]'):

    pred = torch.argmax(logits, dim=1)
    acc = torch.eq(pred, labels).to(torch.float32).mean().item()

    logger.info('{} Loss: {:.3f} | Accuracy: {:.3f} | ({:.2f} s)'.format(header, loss_scalar, acc, time.time()-start))

def vae_forward_pass(vae_model, features, sample_z=False):

    with torch.no_grad():  # Don't modify VAE
        try:
            latent_stats = vae_model.encoder(features)
        except AttributeError:
            latent_stats = vae_model.module.encoder(features)
        if sample_z is False:  # Take mean of latent dist
            z = latent_stats['continuous'][0]
        else:  # Sample from latent dist
            try:
                z =  vae_model.reparameterize(latent_stats)
            except AttributeError:
                z = vae_model.module.reparameterize(latent_stats)

    return z



def downstream_train(args, device, logger, train_loader, test_loader, net, n_epochs=8, vae_model=None, sample_z=False, 
               index_features=False, latent_features=False, feature_idx=None, log_interval_p_epoch=2, pca=None, notebook=False):

    logger.info('Starting downstream training')

    log_interval = args.n_data / args.batch_size // log_interval_p_epoch
    assert log_interval > 1, 'Need more time between logs!'

    if vae_model is not None and latent_features is True:
        logger.info('Using latent space features')
        vae_model.train()
    else:
        n_epochs = 3

    test_loader_iter = iter(test_loader)
    start_time = time.time()
    # xentropy_loss_f = nn.BCEWithLogitsLoss()
    xentropy_loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    net.train()

    index_fn = lambda x: x[:, feature_idx] if index_features is True else x


    for epoch in trange(n_epochs, desc='Epoch'):

        if notebook is True:
            tqdm_f = tqdm_notebook
        else:
            tqdm_f = tqdm

        for step, (features, gen_factors) in enumerate(tqdm_f(train_loader, desc='Train'), 0):

            features = features.to(device, dtype=torch.float)
            gen_factors = gen_factors.to(device)
            labels = gen_factors[:,0].long()

            if vae_model is not None and latent_features is True:
                z = vae_forward_pass(vae_model, features, sample_z)
                features = z

            if index_features is True:
                features = features[:,feature_idx]

            if pca is not None:
                with torch.no_grad():
                    features = torch.Tensor(pca.transform(features.cpu())).cuda()

            logits = net(features)
            loss = xentropy_loss_f(input=logits, target=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_interval == 1:
                report(args, logits, labels, loss.item(), start_time, logger)

                try:
                    test_features, test_gen_factors = test_loader_iter.next()
                except StopIteration:
                    test_loader_iter = iter(test_loader)
                    test_features, test_gen_factors = test_loader_iter.next()

                test_features = test_features.to(device, dtype=torch.float)
                test_gen_factors = test_gen_factors.to(device)
                test_labels = test_gen_factors[:,0].long()

                with torch.no_grad():

                    if vae_model is not None and latent_features is True:
                        z = vae_forward_pass(vae_model, test_features, sample_z)
                        test_features = z

                    if index_features is True:
                        test_features = test_features[:, feature_idx]

                    if pca is not None:
                        with torch.no_grad():
                            test_features = torch.Tensor(pca.transform(test_features.cpu())).cuda()

                    test_logits = net(test_features)
                    test_loss = xentropy_loss_f(input=test_logits, target=test_labels)

                report(args, test_logits, test_labels, test_loss.item(), start_time, logger, header='[TEST]')

        logger.info('Done! Time elapsed: {:.3f} s'.format(time.time()-start_time))

def post_logits(dataloader, device, logger, net, vae_model, sample_z=False, index_features=False, latent_features=False, feature_idx=None, 
        pca=None, notebook=False):

    N = len(dataloader.dataset)

    logits = torch.empty(N,2)
    gen_factors_all = torch.empty(N,3)
    latents = torch.empty(N, net.input_dim)

    n = 0

    if notebook is True:
        tqdm_f = tqdm_notebook
    else:
        tqdm_f = tqdm

    with torch.no_grad():

        for idx, (data, gen_factors) in enumerate(tqdm_f(dataloader, desc='Eval'), 0):
            batch_size = data.shape[0]

            data = data.to(device, dtype=torch.float)
            features = data

            if vae_model is not None and latent_features is True:
                z = vae_forward_pass(vae_model, data, sample_z)
                features = z

            if index_features is True:
                features = features[:,feature_idx]

            if pca is not None:
                with torch.no_grad():
                    features = torch.Tensor(pca.transform(features.cpu())).cuda()

            logits[n:n+batch_size] = net(features).cpu()
            gen_factors_all[n:n+batch_size] = gen_factors
            latents[n:n+batch_size] = features

            n += batch_size

    labels = gen_factors_all[:,0]
    acc = torch.mean(torch.eq(torch.argmax(logits, dim=1), labels).float()).item()

    logger.info('Accuracy: {:.3f}'.format(acc))

    return logits, gen_factors_all, latents

def downstream_metrics(args, model, device, logger, train_loader, test_loader, all_loader, latent_features=True, index_features=True, 
                       sample_z=False, leave_out=[1], notebook=False, n_epochs=8):


    logger.info('Omitting dimensions {}'.format(leave_out))

    if not isinstance(leave_out, list):
        leave_out = list(leave_out)

    try:
        feature_idx = list(set(range(model.latent_dim)).difference(leave_out)) if index_features is True else list(range(model.latent_dim))
    except AttributeError:
        feature_idx = list(set(range(model.module.latent_dim)).difference(leave_out)) if index_features is True else list(range(model.module.latent_dim))
    input_dim = len(feature_idx) if latent_features is True else args.input_dim

    simplenet = network.SimpleDense(input_dim=input_dim, n_classes=2, hidden_dim=256)
    simplenet.to(device)

    downstream_train(args, device, logger, train_loader, test_loader, simplenet, vae_model=model, index_features=index_features, feature_idx=feature_idx, 
        sample_z=sample_z, latent_features=latent_features, notebook=notebook, n_epochs=n_epochs)
    logits, gen_factors, latents = post_logits(all_loader, device, logger, simplenet, vae_model=model, index_features=index_features, 
        latent_features=latent_features, feature_idx=feature_idx, sample_z=sample_z, pca=None, notebook=notebook)

    signal_probs = F.softmax(logits, dim=1).numpy()
    y_prob = signal_probs[:,1]
    labels = gen_factors[:,0].numpy()

    df_post = pd.DataFrame(gen_factors.numpy(), columns=['label', 'B_Mbc', 'B_deltaE'])
    df_post['y_prob'] = y_prob

    df = all_loader.dataset.df

    df['y_prob'] = df_post.y_prob.values

    fpr, tpr, thresholds = roc_curve(df._label, df['y_prob'])
    roc_auc = roc_auc_score(df._label, df.y_prob)
    jsd_metric_mbc = helpers.jsd_metric(df_post)
    jsd_metric_dE = helpers.jsd_metric(df_post, variable='dE')
    # df = df.sort_values('y_prob', ascending=False).drop_duplicates(subset=['_B_eventCached_boevtNum'], keep='first')

    return jsd_metric_mbc, jsd_metric_dE, roc_auc, df


def metric_custom(args, model, device, logger, gpu_id=1, sample_latents=True,
    n_gen_factors=3, n_samples=10000, train_loader=None, test_loader=None, storage=None, evaluate_with_mbc=True, notebook=False):
    """
    Compute MIG and downstream metrics on custom dataset.
    """  
    start_time = time.time()
    torch.cuda.set_device(gpu_id)
    
    # Evaluate on full dataset
    all_loader = datasets.get_dataloaders(args.dataset,
                                   batch_size=args.batch_size,
                                   logger=logger,
                                   train=False,
                                   metrics=True,
                                   evaluate=True,
                                   shuffle=False)

    vae = model.to(device)
    N = len(all_loader.dataset)  # number of data samples - don't shuffle
    try:
        K = vae.latent_dim  # number of latent variables
        enc = vae.encoder
    except AttributeError:
        K = vae.module.latent_dim
        enc = vae.module.encoder
    nparams = 2  # params of q_{\phi}(z|x)
    vae.eval()

    logger.info('Computing q(z|x) distributions.')
    qzCx_params = torch.Tensor(N, K, nparams)
    gen_factors = torch.Tensor(N, n_gen_factors)

    n = 0
    pbar = tqdm(total=N)

    with torch.no_grad():
        for x, gen_factors_n in all_loader:
            batch_size = x.size(0)
            x = x.to(device, dtype=torch.float)
            qzCx_params_n = torch.stack(enc(x)['continuous'], dim=2)
            qzCx_params[n:n + batch_size] = qzCx_params_n.data
            gen_factors[n:n + batch_size] = gen_factors_n
            n += batch_size
            pbar.update(batch_size)
    pbar.close()
    
    # Remove labels from generative factors
    gen_factors = gen_factors[:,1:]
    
    if sample_latents is True:
        # Sample from q(z|x) distribution
        qzCx_samples = qzCx_sample(params=qzCx_params)
    else:
        # Use posterior q(z|x) mean
        qzCx_samples = qzCx_params.select(-1,0)
        
    logger.info('Estimating discrete MIG.')
    discrete_metric, top_mi_discrete_normed, top_mi_idx_discrete = estimate_MIG_discrete_custom(qzCx_samples.view(N,K).data.cpu(), gen_factors, n_samples=10000, multidim=bool(len(args.sensitive_latent_idx) > 1))
    logger.info('MIG (discrete): {:.3f} | Time elapsed: {:.2f}s'.format(discrete_metric, time.time() - start_time))
    
    
    """
    Downstream Training
    """
    
    if test_loader is None:
        test_loader = datasets.get_dataloaders(args.dataset,
                                               batch_size=args.batch_size,
                                               logger=logger,
                                               train=False,
                                               sampling_bias=args.sampling_bias,
                                               shuffle=args.shuffle)

    if train_loader is None:
        train_loader = datasets.get_dataloaders(args.dataset,
                                                batch_size=args.batch_size,
                                                logger=logger,
                                                train=True,
                                                sampling_bias=args.sampling_bias,
                                                shuffle=args.shuffle)
    
    
    # With Mbc dimension. 
    if evaluate_with_mbc is True:
        jsd_mbc, jsd_dE, auc_mbc, df_mbc = downstream_metrics(args, model, device, logger, train_loader, test_loader, all_loader, index_features=False, notebook=notebook, n_epochs=4)
    else:
        jsd_mbc, jsd_dE_mbc, auc_mbc = None, None, None
    
    # Omit Mbc dimension. Supervised: constrained dimension. Unsupervised: Dimension with highest MI w/ Mbc.
    if args.supervision is False:
        omit_idx = int(top_mi_idx_discrete)
        logger.info('Omitting latent dimension {}'.format(str(omit_idx)))
    else:
        omit_idx = args.sensitive_latent_idx
    jsd, jsd_dE, auc, df = downstream_metrics(args, model, device, logger, train_loader, test_loader, all_loader, index_features=True, leave_out=omit_idx, notebook=notebook)

    
    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    args_d.pop('DATASETS'); args_d.pop('LOSSES')
    metrics_out = {'MIG_discrete': discrete_metric, 'JSD_metric_incomplete': jsd, 'JSD_metric_dE_incomplete': jsd_dE, 'AUC_incomplete': auc, 'JSD_metric_complete': jsd_mbc, 
            'JSD_metric_dE_complete': jsd_dE_mbc, 'AUC_complete': auc_mbc, 'args': args_d, 'ckpt': args.ckpt, 'storage': storage}

    save_path = 'disentanglement_metric_custom_{}_{:%Y_%m_%d_%H:%M}.log'.format(args.name, datetime.datetime.now())
    # df_mbc.to_hdf(save_path.replace('.log', '.hdf'), key='df')
    df.to_hdf(os.path.join('results', save_path.replace('.log', '.hdf')), key='df')
    logger.info('Saving to {}'.format(save_path))
    logger.info(metrics_out)

    try:
        helpers.save_metadata(metrics_out, args.output_directory, filename=save_path)
    except AttributeError:
        helpers.save_metadata(metrics_out, 'results', filename=save_path)
    
    if evaluate_with_mbc is True:
        return metrics_out, df_mbc, df
    else:
        return metrics_out, df


def match_z_v(marginal_entropies, conditional_entropies, top_mi_discrete_normed, top_mi_discrete_idx):
    """
    Matches latent dimension z_j with generative factor v_k with the highest
    estimated mutual information I(z_j; v_k) using dSprites.
    
    Returns dictionary containing results of matching for both continuous and
    discrete MI approximations.
    """
    
    factor_entropies = [3, 6, 40, 32, 32]
    mi_info = lambda x: dict(latent_dim=int(top_mi_idx[x]), 
                              mi_normed=float(top_mi_normed[x]), 
                              mi_normed_gap=float(mi_sorted_normed_gap[x].item()),  # normalized gap b/w highest and second highest MI, dimwise
                              discrete_latent_dim=int(top_mi_discrete_idx[x]),
                              discrete_mi_normed=float(top_mi_discrete_normed[x]))

    mutual_infos = torch.unsqueeze(marginal_entropies, dim=0) - conditional_entropies
    mi_sorted, mi_sorted_idx = torch.sort(mutual_infos.clamp(min=0), dim=1, descending=True)
    mi_sorted_normed = mi_sorted / torch.Tensor(factor_entropies).log()[:, None]
    
    mi_sorted_normed_gap = mi_sorted_normed[:,0] - mi_sorted_normed[:,1]

    top_mi_normed = mi_sorted_normed[:,0].numpy()
    top_mi_idx = mi_sorted_idx[:,0].numpy()
    
    z_v_mapping = dict(shape=mi_info(0),
                       scale=mi_info(1),
                       rotation=mi_info(2),
                       posX=mi_info(3),
                       posY=mi_info(4))
    
    return z_v_mapping


def mutual_info_metric_shapes(args, model, device, logger, storage=None, gpu_id=1, sample_latents=True,
        n_gen_factors=5, n_samples=10000):

    """
    Compute MIG on dSprites dataset assuming the posterior distribution
    q(z|x) is Gaussian with diagonal covariance matrix. Uses sklearn estimator 
    to compute discrete mutual information by default.
    """
    start_time = time.time()
    torch.cuda.set_device(gpu_id)

    # Evaluate on full dataset
    all_loader = datasets.get_dataloaders(
                                args.dataset,
                                batch_size=args.batch_size,
                                logger=logger,
                                metrics=True,
                                evaluate=True,
                                train=False,
                                shuffle=False)

    vae = model.to(device)
    N = len(all_loader.dataset)  # number of data samples - don't shuffle
    try:
        K = vae.latent_dim  # number of latent variables
        enc = vae.encoder
    except AttributeError:
        K = vae.module.latent_dim
        enc = vae.module.encoder
    nparams = 2  # params of q_{\phi}(z|x)
    vae.eval()

    logger.info('Computing q(z|x) distributions.')
    qzCx_params = torch.Tensor(N, K, nparams)
    gen_factors = torch.Tensor(N, n_gen_factors)

    n = 0
    pbar = tqdm(total=N)
    
    with torch.no_grad():
        for x, gen_factors_n in all_loader:
            batch_size = x.size(0)
            x = x.to(device, dtype=torch.float)
            qzCx_params_n = torch.stack(enc(x)['continuous'], dim=2)
            qzCx_params[n:n + batch_size] = qzCx_params_n.data
            gen_factors[n:n + batch_size] = gen_factors_n
            n += batch_size
            pbar.update(batch_size)
    pbar.close()

    qzCx_params = qzCx_params.view(3, 6, 40, 32, 32, K, nparams).to(device)
    # gen_factors = gen_factors[:,1:]  # treat shapes factor as noise
    
    if sample_latents is True:
        # Sample from q(z|x) distribution
        qzCx_samples = qzCx_sample(params=qzCx_params)
    else:
        # Use posterior q(z|x) mean
        qzCx_samples = qzCx_params.select(-1,0)
        
    logger.info('Estimating discrete MIG.')
    discrete_metric, top_mi_discrete_normed, top_mi_discrete_idx = estimate_MIG_discrete(qzCx_samples.view(N,K).data.cpu(), gen_factors)
    logger.info('MIG (discrete): {:.3f} | Time elapsed: {:.2f}s'.format(discrete_metric, time.time() - start_time))
        
    logger.info('Estimating marginal entropies.')
    # marginal entropies
    marginal_entropies = estimate_entropies(
        qzCx_samples.view(N, K).transpose(0, 1),
        qzCx_params.view(N, K, nparams), device=device)

    marginal_entropies = marginal_entropies.cpu()
    cond_entropies = torch.zeros(5, K)

    logger.info('Estimating conditional entropies H(z | v)')
    
    logger.info('Estimating conditional entropies for shape.')

    for i in range(3):
        qzCx_samples_i = qzCx_samples[i, :, :, :, :, :].contiguous()
        qzCx_params_i = qzCx_params[i, :, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qzCx_samples_i.view(N // 3, K).transpose(0, 1),
            qzCx_params_i.view(N // 3, K, nparams), device=device)
        
        cond_entropies[0] += cond_entropies_i.cpu() / 3


    logger.info('Estimating conditional entropies for scale.')
    for i in range(6):
        qzCx_samples_i = qzCx_samples[:, i, :, :, :, :].contiguous()
        qzCx_params_i = qzCx_params[:, i, :, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qzCx_samples_i.view(N // 6, K).transpose(0, 1),
            qzCx_params_i.view(N // 6, K, nparams), device=device)
        
        cond_entropies[1] += cond_entropies_i.cpu() / 6

    logger.info('Estimating conditional entropies for orientation.')
    for i in range(40):
        qzCx_samples_i = qzCx_samples[:, :, i, :, :, :].contiguous()
        qzCx_params_i = qzCx_params[:, :, i, :, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qzCx_samples_i.view(N // 40, K).transpose(0, 1),
            qzCx_params_i.view(N // 40, K, nparams), device=device)

        cond_entropies[2] += cond_entropies_i.cpu() / 40

    logger.info('Estimating conditional entropies for pos x.')
    for i in range(32):
        qzCx_samples_i = qzCx_samples[:, :, :, i, :, :].contiguous()
        qzCx_params_i = qzCx_params[:, :, :, i, :, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qzCx_samples_i.view(N // 32, K).transpose(0, 1),
            qzCx_params_i.view(N // 32, K, nparams), device=device)

        cond_entropies[3] += cond_entropies_i.cpu() / 32

    logger.info('Estimating conditional entropies for pox y.')
    for i in range(32):
        qzCx_samples_i = qzCx_samples[:, :, :, :, i, :].contiguous()
        qzCx_params_i = qzCx_params[:, :, :, :, i, :].contiguous()

        cond_entropies_i = estimate_entropies(
            qzCx_samples_i.view(N // 32, K).transpose(0, 1),
            qzCx_params_i.view(N // 32, K, nparams), device=device)
        
        cond_entropies[4] += cond_entropies_i.cpu() / 32
    
    metric, dimwise_MI = compute_metric_shapes(marginal_entropies, cond_entropies[1:])  # omit shapes conditional entropy
    logger.info('MIG: {:.3f} | MIG (discrete): {:.3f} | Time elapsed: {:.2f}s'.format(metric, discrete_metric, time.time() - start_time))
    
    z_v_mapping = match_z_v(marginal_entropies, cond_entropies, top_mi_discrete_normed, top_mi_discrete_idx)

    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    metrics_out = {'MIG': metric.item(), 'MIG_discrete': discrete_metric, 'args': args_d, 'ckpt': args.ckpt, 'z_v_mapping': z_v_mapping, 'storage': storage}
    save_path = 'disentanglement_metric_{}_{:%Y_%m_%d_%H:%M}.log'.format(args.name, datetime.datetime.now())
    logger.info('Saving to {}'.format(save_path))
    logger.info(metrics_out)

    try:
        helpers.save_metadata(metrics_out, args.output_directory, filename=save_path)
    except AttributeError:
        helpers.save_metadata(metrics_out, 'results', filename=save_path)

    return metric, discrete_metric, marginal_entropies, cond_entropies, z_v_mapping

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save', type=str, default='results')
    parser.add_argument('--n_samples', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='dsprites', choices=['dsprites', 'custom'])
    parser.add_argument('--complete_eval', action='store_true', help='Train downstream model with full representation as well.')
    cmd_args = parser.parse_args()

    if cmd_args.gpu != 0:
        torch.cuda.set_device(cmd_args.gpu)
        
    start_time = time.time()
    device = helpers.get_device()

    args, model = helpers.load_model(cmd_args.ckpt, device)
    args.ckpt = cmd_args.ckpt
    logger = helpers.logger_setup()
    logger.info('Using {} samples to estimate entropies.'.format(cmd_args.n_samples))

    if args.dataset == 'dsprites':
        metric, discrete_metric, H_z, H_zCv, z_v_mapping = mutual_info_metric_shapes(args, model, device, logger, n_samples=cmd_args.n_samples)
        logger.info(z_v_mapping)
        logger.info('MIG: {:.3f} | Time elapsed: {:.2f}s'.format(metric, time.time() - start_time))
        logger.info('MIG (discrete): {:.3f} | Time elapsed: {:.2f}s'.format(discrete_metric, time.time() - start_time))

        save_path_pt = 'disentanglement_metric_{}_{}_{:%Y_%m_%d_%H:%M}.pth'.format(args.name, args.dataset, datetime.datetime.now())

        torch.save({
            'MIG': metric,
            'MIG_discrete': discrete_metric,
            'marginal_entropies': H_z,
            'cond_entropies': H_zCv,
            'z_v_mapping': z_v_mapping,
            'args': args,
            'ckpt': cmd_args.ckpt
            }, os.path.join(cmd_args.save, save_path_pt))

    elif args.dataset == 'custom':
        metrics, *_ = metric_custom(args, model, device, logger, evaluate_with_mbc=cmd_args.complete_eval)
        logger.info(metrics)
        logger.info('Time elapsed: {:.2f}s'.format(time.time() - start_time))
    else:
        raise NotImplementedError

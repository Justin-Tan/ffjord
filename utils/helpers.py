""" Authors: @YannDubs 2019
             @sksq96   2019 """

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
import os, time, datetime
import logging

from scipy.stats import entropy
from collections import OrderedDict
from sklearn.metrics import mutual_info_score

from models import network, vae
from utils import distributions

META_FILENAME = "specs.json"

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def get_device(is_gpu=True):
    """Return the correct device"""
    return torch.device("cuda" if torch.cuda.is_available() and is_gpu
                        else "cpu")

def get_model_device(model):
    """Return the device on which a model is."""
    return next(model.parameters()).device


def makedirs(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def setup_signature(args):

    time_signature = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now()).replace(':', '_')
    args.name = '{}_{}_{}_{}'.format(args.name, args.dataset, args.loss_type, time_signature)

    if args.flow != 'no_flow':
        args.name = '{}_{}'.format(args.name, args.flow)

    if args.loss_type == 'beta_VAE':
        args.name = '{}_beta{}'.format(args.name, args.beta)
    elif args.loss_type == 'annealed_VAE':
        args.name = '{}_gamma{}'.format(args.name, args.gamma)
    elif args.loss_type == 'factor_VAE':
        args.name = '{}_gamma_fvae{}'.format(args.name, args.gamma_fvae)
    elif 'TCVAE' in args.loss_type:
        args.name = '{}_betatcvae{}'.format(args.name, args.beta_btcvae)

    if args.supervision is True:
        args.name = '{}_lambda{}'.format(args.name, args.supervision_lagrange_m)
        args.name = '{}_sidx{}'.format(args.name, str(args.sensitive_latent_idx))

    args.snapshot = os.path.join(args.save, args.name)
    makedirs(args.snapshot)
    args.checkpoints_save = os.path.join(args.snapshot, 'checkpoints')

    return args

def save_metadata(metadata, directory='results', filename=META_FILENAME, **kwargs):
    """ Save the metadata of a training directory.
    Parameters
    ----------
    metadata:
        Object to save
    directory: string
        Path to folder where to save model. For example './experiments/mnist'.
    kwargs:
        Additional arguments to `json.dump`
    """
    path_to_metadata = os.path.join(directory, filename)

    with open(path_to_metadata, 'w') as f:
        json.dump(metadata, f, indent=4, sort_keys=True)  #, **kwargs)

def save_model(model, optimizer, mean_loss, directory, epoch, device, args,
               multigpu=False):
 
    makedirs(directory)
    model.cpu()  # Move model parameters to CPU for consistency when restoring

    metadata = dict(input_dim=args.input_dim, latent_dim=args.latent_dim,
                    model_loss=args.loss_type)

    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    metadata.update(args_d)
    args_d['timestamp'] = '{:%Y_%m_%d_%H:%M}'.format(datetime.datetime.now())
    
    model_name = args.name
    metadata_path = os.path.join(directory, 'metadata/model_{}_metadata_{:%Y_%m_%d_%H:%M}.json'.format(model_name, datetime.datetime.now()))

    if not os.path.exists(os.path.join(directory, 'metadata')):
        os.makedirs(os.path.join(directory, 'metadata'))
    
    if not os.path.isfile(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)
            
    model_path = os.path.join(directory, 'model_{}_epoch_{}_{:%Y_%m_%d_%H:%M}.pt'.format(model_name, epoch, datetime.datetime.now()))

    if os.path.exists(model_path):
        model_path = os.path.join(directory, 'model_{}_epoch_{}_{:%Y_%m_%d_%H:%M:%S}.pt'.format(model_name, epoch, datetime.datetime.now()))

    torch.save({'model_state_dict': model.module.state_dict() if args.multigpu is True else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'mean_epoch_loss': mean_loss,
                'args': args_d
                }, f=model_path)
    
    model.to(device)  # Move back to device

    print('Model saved to path {}'.format(model_path))
    return model_path
   

def save_model_online(model, optimizer, epoch, save_dir, name):
    save_path = os.path.join(save_dir, '{}_epoch{}.pt'.format(name, epoch))
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
    print('Model saved to path {}'.format(save_path))
    
def load_model(save_path, device, optimizer=None, prediction=True):
    checkpoint = torch.load(save_path)
    args = checkpoint['args']
    args = Struct(**args)

    # Backwards compatibility
    if args.dataset == 'custom':
        args.custom = True
    if hasattr(args, 'sampling_bias') is False:
        args.sampling_bias = False
    if hasattr(args, 'flow_hidden_dim') is False:
        args.flow_hidden_dim = 36

    try:
        if args.use_flow is True:
            model = vae.realNVP_VAE(args)
        else:
            model = vae.VAE(args)
    except AttributeError:
        model = vae.VAE(args)


    model.load_state_dict(checkpoint['model_state_dict'])

    if prediction:
        model.eval()
    else:
        model.train()
        
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return args, model.to(device), optimizer
    
    # Backwards compatibility
    if hasattr(args, 'sampling_bias') is False:
        args.sampling_bias = False

    return args, model.to(device)

def logger_setup_alpha():
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())
    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


def logger_setup(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    # with open(filepath, "r") as f:
    #     logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger

def log(storage, epoch, counter, mean_epoch_loss, total_loss, best_loss, start_time, epoch_start_time, 
        batch_size, header='[TRAIN]', log_interval=100, logger=None):
    
    improved = ''
    t0 = epoch_start_time
    
    if total_loss < best_loss:
        best_loss = total_loss
        improved = '[*]'  
    
    storage['epoch'].append(epoch)
    storage['mean_epoch_loss'].append(mean_epoch_loss)
    storage['time'].append(time.time())

    try:
        reconstruction_loss = storage['reconstruction_loss'][-1]
        kl_loss = storage['kl_loss'][-1]
        elbo = storage['ELBO'][-1]
    except IndexError:
        reconstruction_loss, kl_loss, elbo = np.nan, np.nan, np.nan

    if logger is not None:
        report_f = logger.info   
    else:
        report_f = print

    report_f(header)

    if header == '[TRAIN]':
        report_f("Epoch {} | Mean epoch loss: {:.3f} | Total loss: {:.3f} | ELBO: {:.3f} | Reco Loss: {:.3f} | KL Loss: {:.3f} | "
                 "Rate: {} examples/s | Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, total_loss, elbo, 
                 reconstruction_loss, kl_loss, int(batch_size*counter / (log_interval * (time.time()-t0))), time.time()-start_time, improved))
    else:
        report_f("Epoch {} | Mean epoch loss: {:.3f} | Total loss: {:.3f} | ELBO: {:.3f} | Reco Loss: {:.3f} | KL Loss: {:.3f} | "
                 "Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, total_loss, elbo, reconstruction_loss,
                 kl_loss, time.time()-start_time, improved))

    return best_loss


def summary(model, input_size, batch_size=-1, device="cuda"):

    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            elif isinstance(output, dict):
                summary[m_key]["output_shape"] = []
                getsize = lambda x: [-1] + [int(np.squeeze(list(o.size())[1:])) for o in x]
                # output.pop('hidden')
                output = {}
                for k, v in zip(output.keys(), output.values()):
                    if isinstance(out, list):
                        output_i = {k: torch.cat(v, axis=-1)}
                    elif isinstance(out, torch.Tensor):
                        output_i = {k: v}
                    output.update(output_i)

                summary[m_key]["output_shape"] += [getsize(output.values())]

                # summary[m_key]["output_shape"] = [
                #    [-1] + list(v.size())[1:] for v in output.values()]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0

    flatten = lambda x: [elem for sl in x for elem in sl] 

    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        os = summary[layer]["output_shape"]
        flat1 = [i for i in os if not isinstance(i, list)]
        flat2 = flatten([i for i in os if isinstance(i, list)])
        os = flat1 + flat2
        total_output += np.prod(os)
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")

def jsd_metric(df, selection_fraction=0.005, nbins=32, dE_min=-0.25, dE_max=0.1, mbc_min=5.2425, mbc_max=5.29, variable='B_Mbc'):
    """
    Attempt to quantify sculpting.
    Evaluates mass decorrelation on some blackbox learner by evaluating a discrete
    approximation of the Jensen-Shannon divergence between the distributions of interest
    (here a mass-related quantity) passing and failing some learner threshold. If the 
    learned representation used for classification is noninformative of the variable of
    interest this should be low.
    """

    def _one_hot_encoding(x, nbins):
        x_one_hot = np.zeros((x.shape[0], nbins))
        # x_one_hot[np.arange(x.shape[0]), x] = 1
        x_one_hot[np.arange(x.shape[0]), np.max(x, nbins-1)] = 1
        x_one_hot_sum = np.sum(x_one_hot, axis=0)/x_one_hot.shape[0]

        return x_one_hot_sum

    try:
        df_bkg = df[df.label<0.5]
    except AttributeError:
        df_bkg = df[df.y_true<0.5]

    #try:
    #    df_bkg = df_bkg[df_bkg.B_deltaE > -0.25].query('B_deltaE < 0.1')
    #except AttributeError:
    #    df_bkg = df_bkg[df_bkg._B_deltaE > -0.25].query('_B_deltaE < 0.1')

    select_bkg = df_bkg.nlargest(int(df_bkg.shape[0]*selection_fraction), columns='y_prob')
    print('Surviving events', select_bkg.shape)
    min_threshold = select_bkg.y_prob.min()
    print(min_threshold)

    df_pass = df_bkg[df_bkg.y_prob > min_threshold]
    df_fail = df_bkg[df_bkg.y_prob < min_threshold]
    print('Passing events', df_pass.shape)

    try:
        df_bkg_pass = df_pass[df_pass.label < 0.5]
        df_bkg_fail = df_fail[df_fail.label < 0.5]
    except AttributeError:
        df_bkg_pass = df_pass[df_pass.y_true < 0.5]
        df_bkg_fail = df_fail[df_fail.y_true < 0.5]
    print('Passing bkg events', df_bkg_pass.shape)

    # N_bkg_pass = int(df_bkg_pass._weight_.sum())
    # N_bkg_fail = int(df_bkg_fail._weight_.sum())
    # print('N_bkg_pass / N_bkg_fail: {}'.format(N_bkg_pass/N_bkg_fail))

    # Discretization
    if variable == 'B_Mbc':
        try:
            bkg_pass_discrete = np.digitize(df_bkg_pass.B_Mbc, bins=np.linspace(mbc_min,mbc_max,nbins+1), right=False)-1
            bkg_fail_discrete = np.digitize(df_bkg_fail.B_Mbc, bins=np.linspace(mbc_min,mbc_max,nbins+1), right=False)-1
        except AttributeError:
            bkg_pass_discrete = np.digitize(df_bkg_pass._B_Mbc, bins=np.linspace(mbc_min,mbc_max,nbins+1), right=False)-1
            bkg_fail_discrete = np.digitize(df_bkg_fail._B_Mbc, bins=np.linspace(mbc_min,mbc_max,nbins+1), right=False)-1
    elif variable =='dE':
        try:
            bkg_pass_discrete = np.digitize(df_bkg_pass.B_deltaE, bins=np.linspace(dE_min,dE_max,nbins+1), right=False)-1
            bkg_fail_discrete = np.digitize(df_bkg_fail.B_deltaE, bins=np.linspace(dE_min,dE_max,nbins+1), right=False)-1
        except AttributeError:
            bkg_pass_discrete = np.digitize(df_bkg_pass._B_deltaE, bins=np.linspace(dE_min,dE_max,nbins+1), right=False)-1
            bkg_fail_discrete = np.digitize(df_bkg_fail._B_deltaE, bins=np.linspace(dE_min,dE_max,nbins+1), right=False)-1

    bkg_pass_sum = _one_hot_encoding(bkg_pass_discrete, nbins)
    bkg_fail_sum = _one_hot_encoding(bkg_fail_discrete, nbins)

    M = 0.5*bkg_pass_sum + 0.5*bkg_fail_sum

    kld_pass = entropy(bkg_pass_sum, M)
    kld_fail = entropy(bkg_fail_sum, M)

    jsd_discrete = 0.5*kld_pass + 0.5*kld_fail

    return jsd_discrete

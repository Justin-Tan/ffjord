""" Authors: @YannDubs 2019
             @sksq96   2019 """

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np
import json
import os, time, datetime
import logging
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

def save_metadata(metadata, directory, filename=META_FILENAME, **kwargs):
    """Load the metadata of a training directory.
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
 
    model.cpu()  # Move model parameters to CPU for consistency when restoring

    metadata = dict(im_shape=args.im_shape, latent_dim=args.latent_dim,
                    model_loss=args.loss_type)

    args_d = dict((n, getattr(args, n)) for n in dir(args) if not (n.startswith('__') or 'logger' in n))
    metadata.update(args_d)
    
    model_name = args.name
    metadata_path = os.path.join(directory, 'metadata/model_{}_metadata_{:%Y_%m_%d_%H:%M}.json'.format(model_name, datetime.datetime.now()))

    if not os.path.exists(os.path.join(directory, 'metadata')):
        os.makedirs(os.path.join(directory, 'metadata'))
    
    if not os.path.isfile(metadata_path):
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4, sort_keys=True)
            
    model_path = os.path.join(directory, 'model_{}_epoch_{}_{:%Y_%m_%d_%H:%M}.pt'.format(model_name, epoch, datetime.datetime.now()))
    torch.save({'model_state_dict': model.module.state_dict() if args.multigpu is True else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'mean_epoch_loss': mean_loss,
                'args': args_d
                }, f=model_path)
    
    model.to(device)  # Move back to device
    print('Model saved to path {}'.format(model_path))
   

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

    if not hasattr(args, 'prior') or args.prior == 'normal':
        prior = distributions.Normal()
    elif args.prior == 'flow':
        prior = distributions.FactorialNormalizingFlow(dim=args.latent_dim, nsteps=args.flow_steps)

    x_dist = distributions.Bernoulli()
    
    model = vae.VAE(args, latent_spec=args.latent_spec, prior=prior, x_dist=x_dist)
    model.load_state_dict(checkpoint['model_state_dict'])

    if prediction:
        model.eval()
    else:
        model.train()
        
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return args, model.to(device), optimizer
    
    return args, model.to(device)

def logger_setup():
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s: %(message)s', 
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel('INFO'.upper())
    stream = logging.StreamHandler()
    stream.setLevel('INFO'.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)
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
    except IndexError:
        reconstruction_loss, kl_loss = np.nan, np.nan
    
    print(header)

    if header == '[TRAIN]':
        print("Epoch {} | Mean epoch loss: {:.3f} | Total loss: {:.3f} | Reco Loss: {:.3f} | KL Loss: {:.3f} | "
            "Rate: {} examples/s | Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, total_loss, reconstruction_loss,
            kl_loss, int(batch_size*counter / (log_interval * (time.time()-t0))), time.time()-start_time, improved))
    else:
        print("Epoch {} | Mean epoch loss: {:.3f} | Total loss: {:.3f} | Reco Loss: {:.3f} | KL Loss: {:.3f} | "
            "Time: {:.1f} s | Improved: {}".format(epoch, mean_epoch_loss, total_loss, reconstruction_loss,
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
                summary[m_key]["output_shape"] += [getsize(v) for v in output.values()]

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

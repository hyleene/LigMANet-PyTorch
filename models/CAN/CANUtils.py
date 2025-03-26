import h5py
import torch
import shutil
import numpy as np
import os
from datetime import date

def save_net(fname, net):
    """ Saves the network

        :param string fname: file name of the saved model
        :param Object net: model to be saved
    """
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())

def load_net(fname, net):
    """ Loads the network
    
        :param string fname: file name of the network to be loaded
        :param Object net: network to be loaded
    """
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    """ Saves the current model progress as a checkpoint
    
        :param dictionary state: current model state
        :param boolean is_best: whether the current model is the best
        :param string save_path: path to relevant files
        :param string filename: filename of checkpoint
    """
    save_path = os.path.join(save_path, filename)
    torch.save(state, save_path)
    if is_best:
        shutil.copyfile(save_path, save_path.replace('checkpoint', 'model_best'))                   

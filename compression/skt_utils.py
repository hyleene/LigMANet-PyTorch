'''
    Code taken directly from the official SKT github repository
    https://github.com/chen-judge/SKT

    found in utils.py
'''

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import os
import time
import numpy as np

class AverageMeter(object):
    def __init__(self):
        """ Initializes an AverageMeter object
        """
        self.reset()

    def reset(self):
        """ Resets the attributes of the AverageMeter object
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Updates the values of the AverageMeter object
        
            :param int val: new value of the object
            :param int n: amount to be added to the count of the object
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_net(fname, net):
    """ Saves the network
    
        :param string fname: file name
        :param Object net: network to be saved
    """
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    """ Loads the specified network
    
        :param string fname: file name
        :param Object net: network to be loaded
    """
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            

def save_checkpoint(state, mae_is_best, mse_is_best, path, save_all=False, filename='checkpoint.pth.tar'):
    """ Saves the current training epoch as a checkpoint
    
        :param dictionary state: current state of the model
        :param boolean mae_is_best: whether the MAE of the current epoch is the lowest
        :param boolean mse_is_best: whether the RMSE of the current epoch is the lowest
        :param string path: path to the folder where the checkpoint is saved
        :param boolean save_all: whether all epochs are to be saved
        :param string filename: file name of the checkpoint
    """
    torch.save(state, os.path.join(path, filename))
    epoch = state['epoch']
    if save_all:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'saveall_epoch'+str(epoch)+".pth.tar"))
        return
    if mae_is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'epoch'+str(epoch)+'_best_mae.pth.tar'))
    if mse_is_best:
        shutil.copyfile(os.path.join(path, filename), os.path.join(path, 'epoch'+str(epoch)+'_best_mse.pth.tar'))


def cal_para(net):
    """ Calculates the number of parameters of the model
    
        :param Object net: model whose parameters are to be calculated
    """
    params = list(net.parameters())
    k = 0
    for i in params:
        l = 1
        # print "stucture of layer: " + str(list(i.size()))
        for j in i.size():
            l *= j
        # print "para in this layer: " + str(l)
        k = k + l
    print("the amount of para: " + str(k))


def crop_img_patches(img, size=512):
    """ Crops the test images to patches
    
        :param Object img: image to be cropped
        :param int size: maximum size of the cropped images

        :returns: Cropped images

        :rtype: list
    """
    
    ''' While testing UCF data, we load original images, then use crop_img_patches to crop the test images to patches,
        calculate the crowd count respectively and sum them together finally. 
    '''
    w = img.shape[3]
    h = img.shape[2]
    x = int(w/size)+1
    y = int(h/size)+1
    crop_w = int(w/x)
    crop_h = int(h/y)
    patches = []
    for i in range(x):
        for j in range(y):
            start_x = crop_w*i
            if i == x-1:
                end_x = w
            else:
                end_x = crop_w*(i+1)

            start_y = crop_h*j
            if j == y - 1:
                end_y = h
            else:
                end_y = crop_h*(j+1)

            sub_img = img[:, :, start_y:end_y, start_x:end_x]
            patches.append(sub_img)
    return patches

def cosine_similarity(stu_map, tea_map):
    """ Calculates the cosine similarity between the student and teacher feature maps
    
        :param torch.Tensor stu_map: student feature map
        :param torch.Tensor tea_map: teacher feature map

        :returns: quantification of difference between the student and teacher feature maps

        :rtype: torch.Tensor
    """
    similiar = 1-F.cosine_similarity(stu_map, tea_map, dim=1)
    loss = similiar.sum()
    return loss

def cal_dense_fsp(features):
    """ Calculates the dense flow of solution procedure among the model features
    
        :param list features: feature values of the model

        :returns: flow of solution procedure matrix among the features

        :rtype: list
    """
    fsp = []
    for groups in features:
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                x = groups[i]
                y = groups[j]

                norm1 = nn.InstanceNorm2d(x.shape[1])
                norm2 = nn.InstanceNorm2d(y.shape[1])
                x = norm1(x)
                y = norm2(y)
                res = gram(x, y)
                fsp.append(res)
    return fsp

def scale_process(features, scale=[3, 2, 1], ceil_mode=True):
    """ Processes features for multi-scale dense FSPs
    
        :param list features: feature values of the model
        :param list scale: scales of the model
        :param boolean ceil_mode: whether ceil is used to calculate the output shape

        :returns: Updated feature values of the model

        :rtype: list
    """
    new_features = []
    for i in range(len(features)):
        if i >= len(scale):
            new_features.append(features[i])
            continue
        down_ratio = pow(2, scale[i])
        pool = nn.MaxPool2d(kernel_size=down_ratio, stride=down_ratio, ceil_mode=ceil_mode)
        if isinstance(features[i], list):
            new_features.append(pool(features[i][0]))
        else:
            new_features.append(pool(features[i]))
    return new_features

def scale_process_CAN(features, scale=[3, 2, 1], ceil_mode=True):
    """ Processes features for multi-scale dense FSPs for CAN
    
        :param list features: feature values of the model
        :param list scale: scales of the model
        :param boolean ceil_mode: whether ceil is used to calculate the output shape

        :returns: Updated feature values of the model

        :rtype: list
    """
    new_features = []
    for i in range(len(features)):
        if i >= len(scale):
            new_features.append(features[i])
            continue
        down_ratio = pow(2, scale[i])
        pool = nn.MaxPool2d(kernel_size=down_ratio, stride=down_ratio, ceil_mode=ceil_mode)
        new_features.append(pool(features[i]))
    return new_features


def gram(x, y):
    """ Helper function for computing the FSP matrix
    
        :param torch.Tensor x: first feature
        :param torch.Tensor y: second feature

        :returns: Result of the matrix multiplication of x and y

        :rtype: torch.Tensor
    """
    n = x.shape[0]
    c1 = x.shape[1]
    c2 = y.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    x = x.view(n, c1, -1, 1)[0, :, :, 0]
    y = y.view(n, c2, -1, 1)[0, :, :, 0]
    y = y.transpose(0, 1)
    # print x.shape
    # print y.shape
    z = torch.mm(x, y) / (w*h)
    return z
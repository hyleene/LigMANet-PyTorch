import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import h5py
import cv2
import glob

class listDataset(Dataset):
    def __init__(self, config, root, shape=None, transform=None,  train=False, seen=0,
                 batch_size=1, num_workers=20):
        """ Initializes a listDataset object
        
        Arguments:
            config {Object} -- configurations of the model
            root {string} -- path to root directory of the dataset images
        
        Keyword Arguments:
            shape {list} -- shape of the dataset {default: None}
            transform {Object} -- transform operation perfoemd on the dataset images {default: None}
            train {boolean} -- whether the data will be used for model training {default: False}
            seen {int} -- number of input images seen {default: 0}
            batch_size {int} -- number of samples processed per batch {default: 1}
            num_workers {int} -- number of subprocesses used for data loading {default: 20}
        """
        # if train and dataset == 'shanghai':
        #     root = root*4
        # random.shuffle(root)
        
        # self.nSamples = len(root)
        self.lines = []
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers

        root = str(root)
        self.root = root

        if config.dataset == 'UCFCC50':
            buffer = []
            for i in range(1, 5):
                if i == config.cc50_val:
                    self.val_lines = glob.glob(os.path.join(root, 'fold_'+str(i), 'images', '*.jpg'))
                else:
                    buffer.append(glob.glob(os.path.join(root, 'fold_'+str(i), 'images', '*.jpg')))
            
            for item in buffer:
                for x in item:
                    self.lines.append(x)
            self.nSamples = len(self.lines)
        else:
            self.lines = glob.glob(os.path.join(root, 'train', 'images', '*.jpg'))
            self.val_lines = glob.glob(os.path.join(root, 'images', '*.jpg').replace('train', 'val'))
            self.nSamples = len(self.lines)

    def __len__(self):
        """ Gets the length of the dataset
        
        Returns:
            int -- number of images in the dataset
        """
        return self.nSamples

    def __getitem__(self, index):
        """ Retrieves the item at the specified index
        
        Arguments:
            index {int} -- index of item to be retrieved
        
        Returns:
            Image -- image at the specified index
            np.array -- ground truth density map of the image
        """
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]

        img, target = load_data(img_path, self.train, self.root)

        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
def load_data(img_path,train=True, dataset='Shanghaitech-A'):
    """ Loads the image data
        
        :param string img_path: path to the dataset images
        :param boolean train: whether the dataset is used for training
        :param string dataset: name of dataset to be used

        :returns:
            - (:py:class:`Image`) - retrieved image
            - (:py:class:`np.array`) - ground truth density map of the image
    """
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'density_maps')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if train:
        # if dataset == 'Shanghaitech-A':
        #     crop_ratio = random.uniform(0.5, 1.0)
        #     crop_size = (int(crop_ratio*img.size[0]), int(crop_ratio*img.size[1]))
        #     dx = int(random.random() * (img.size[0]-crop_size[0]))
        #     dy = int(random.random() * (img.size[1]-crop_size[1]))

        #     img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        #     target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]

        # Use crop_ratio between 0.5 and 1.0 for random crop
        ratio = 0.5
        crop_size = (int(img.size[0]*ratio),int(img.size[1]*ratio))
        rdn_value = random.random()

        if rdn_value<0.25:
            dx = 0
            dy = 0
        elif rdn_value<0.5:
            dx = int(img.size[0]*ratio)
            dy = 0
        elif rdn_value<0.75:
            dx = 0
            dy = int(img.size[1]*ratio)
        else:
            dx = int(img.size[0]*ratio)
            dy = int(img.size[1]*ratio)
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:(crop_size[1]+dy),dx:(crop_size[0]+dx)]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target = reshape_target(target, 3)
    target = np.expand_dims(target, axis=0)

    img = img.copy()
    target = target.copy()
    return img, target

def reshape_target(target, down_sample=3):
    """ Downsamples the ground truth to 1/8 of its original size
    
        :param np.array target: ground truth density map to be downsampled
        :param int down_sample: ratio to downsample the ground truth density map

        :returns: Downsampled ground truth density map

        :rtype: np.array
    """
    height = target.shape[0]
    width = target.shape[1]

    # ceil_mode=True for nn.MaxPool2d in model
    for i in range(down_sample):
        height = int((height+1)/2)
        width = int((width+1)/2)
        # height = int(height/2)
        # width = int(width/2)

    target = cv2.resize(target, (width, height), interpolation=cv2.INTER_CUBIC) * (2**(down_sample*2))
    return target
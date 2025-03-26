from PIL import Image
import torch.utils.data as data
import os
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import re
import numpy as np
from utilities.augmentations import enhance_contrast, save_image

def random_crop(im_h, im_w, crop_h, crop_w):
    """ Performs random cropping on the input image
    
        :param int im_h: height of the input image
        :param int im_w: width of the input image
        :param int crop_h: target height of the cropped image
        :param int crop_w: target width of the cropped image

        :returns:
            - (:py:class:`int`) - random height of the cropped image
            - (:py:class:`int`) - random width of the cropped image 
            - (:py:class:`int`) - target height of the cropped image
            - (:py:class:`int`) - target width of the cropped image
    """
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def cal_innner_area(c_left, c_up, c_right, c_down, bbox):
    """ Calculates the inner area of the image
    
        :param list c_left: left bound of the image crop
        :param list c_up: upper bound of the image crop
        :param list c_right: right bound of the image crop
        :param list c_down: lower bound of the image crop
        :param list bbox: bounding box of the image crop

        :returns: inner area of the image

        :rtype: double
    """
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

class Crowd(data.Dataset):
    def __init__(self, root_path, crop_size,
                 downsample_ratio, 
                 dataset, cc_50_val, cc_50_test, is_gray, augment_contrast, augment_contrast_factor, augment_save_location,
                 augment_save, method):
        """ Initializes a Crowd object
        
        Arguments:
            root_path {string} -- path to the root folder
            crop_size {int} -- crop size of the images
            downsample_ratio {int} -- downsample ratio of the images
            dataset {string} -- name of the dataset to be used
            cc_50_val {int} -- fold number of the validation set for the UCF-CC-50 dataset
            cc_50_test {int} -- fold number of the test set for the UCF-CC-50 dataset
            is_gray {boolean} -- whether the images are in grayscale
            augment_contrast {boolean} -- whether contrast augmentation is applied
            augment_contrast_factor {double} -- contrast factor for image contrast augmentation
            augment_save_location {string} -- path to the folder where the augmented images are saved
            augment_save {boolean} -- whether the augmented images are to be saved
            method {string} -- how the data will be used (for model training, validation, or testing)
        """
        self.contrast = augment_contrast
        self.contrast_factor = augment_contrast_factor
        self.augment_save = augment_save
        self.augment_save_location = augment_save_location
        self.dataset = dataset
        
        if dataset != 'UCFCC50':
            self.root_path = root_path
            self.method = method
            self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
            
            if method not in ['train', 'val', 'test']:
                raise Exception("not implement")
        else:
            new_path = "/".join(root_path.split("/")[0:4])
            self.im_list = []
            self.method = method
            
            if self.method == 'train':
                for i in range(1, 5):
                    if (i != cc_50_val):
                        self.root_path = new_path
                        self.im_list.append(sorted(glob(os.path.join(self.root_path, 'fold_'+str(i), '*.jpg'))))
                self.im_list = [j for sub in self.im_list for j in sub]
            elif self.method == 'val':
                self.root_path = new_path
                self.im_list = sorted(glob(os.path.join(self.root_path, 'fold_'+str(cc_50_val), '*.jpg')))
            else:
                self.root_path = new_path
                self.im_list = sorted(glob(os.path.join(self.root_path, 'fold_'+str(cc_50_test), '*.jpg')))

        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio

        if is_gray:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        """ Gets the length of the dataset
        
        Returns:
            int -- number of images in the dataset
        """
        return len(self.im_list)

    def __getitem__(self, item):
        """ Retrieves the item at the specified index
        
        Arguments:
            index {int} -- index of item to be retrieved
        
        Returns:
            Image -- image at the specified index
            int -- number of keypoints in the image
            string -- name of the image
        """
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        try:
            img = Image.open(img_path).convert('RGB')
        except:
            print(os.path.basename(img_path).split('.')[0])
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints, int(re.search(r'\d+',img_path.split('/')[-1]).group()), gd_path)
        elif self.method == 'val' or self.method == 'test':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints, img_path, gd_path):
        """ Transforms the image for model training
        
            :param Image img: input image to be used
            :param np.array keypoints: ground truth density map of the image
            :param string img_path: path to the input image
            :param string gd_path: path to the ground truth density map of the input image

            :returns:
                - (:py:class:`Image`) - transformed version of the input image
                - (:py:class:`float`) - float representation of the ground truth density map
                - (:py:class:`float`) - float representation of the generated density map
                - (:py:class:`int`) - smaller dimension between the image width and height
        """
        # Perform contrast enhancement if enabled
        if self.contrast == True:
            img = enhance_contrast(img, self.contrast_factor)
            if self.augment_save == True:
                if self.dataset == "UCFCC50":
                    image_append = 30
                else:
                    image_append = 400
                filename = "IMG_"+(str(img_path+image_append))
                save_image(img, self.augment_save_location, filename)
                save_gt(gd_path, self.augment_save_location, filename)
                
        # Random crop image patch and find people in it
        wd, ht = img.size
        assert len(keypoints) > 0
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')
        re_size = random.random() * 0.5 + 0.75
        wdd = (int)(wd*re_size)
        htt = (int)(ht*re_size)
        if min(wdd, htt) >= self.c_size:
            wd = wdd
            ht = htt
            img = img.resize((wd, ht))
            keypoints = keypoints*re_size
        st_size = min(wd, ht)
        assert st_size >= self.c_size
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            nearest_dis = np.clip(keypoints[:, 2], 4.0, 128.0)

            points_left_up = keypoints[:, :2] - nearest_dis[:, None] / 2.0
            points_right_down = keypoints[:, :2] + nearest_dis[:, None] / 2.0
            bbox = np.concatenate((points_left_up, points_right_down), axis=1)
            inner_area = cal_innner_area(j, i, j + w, i + h, bbox)
            origin_area = nearest_dis * nearest_dis
            ratio = np.clip(1.0 * inner_area / origin_area, 0.0, 1.0)
            mask = (ratio >= 0.3)

            target = ratio[mask]
            keypoints = keypoints[mask]
            keypoints = keypoints[:, :2] - [j, i]  # change coodinate
        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            target = np.array([])
            if random.random() > 0.5:
                img = F.hflip(img)
        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), \
               torch.from_numpy(target.copy()).float(), st_size
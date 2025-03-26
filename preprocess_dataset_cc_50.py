from scipy.io import loadmat
from PIL import Image
import numpy as np
import os
from glob import glob
import cv2
import argparse


def cal_new_size(im_h, im_w, min_size, max_size):
    """ Calculates the new size of the input image
    
        :param int im_h: height of the input image
        :param int im_w: width of the input image
        :param int min_size: required minimum size of the input images
        :param int max_size: required maximum size of the input images

        :returns:
            - (:py:class:`int`) - new image height
            - (:py:class:`int`) - new image width
            - (:py:class:`double`) - ratio of new image dimensions to the original image dimensions
    """
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


def find_dis(point):
    """ Finds the distance between a set of points
    
        :param list point: coordinates of the set of points

        :returns: Distance between the specified points

        :rtype: double
    """
    square = np.sum(point*points, axis=1)
    dis = np.sqrt(np.maximum(square[:, None] - 2*np.matmul(point, point.T) + square[None, :], 0.0))
    dis = np.mean(np.partition(dis, 3, axis=1)[:, 1:4], axis=1, keepdims=True)
    return dis

def generate_data(im_path):
    """ Extracts the relevant data from the input image
    
        :param string im_path: path to the input image

        :returns:
            - (:py:class:`Image`) - input image
            - (:py:class:`list`) - coordinates of head locations on the density map
    """
    im = Image.open(im_path)
    im_w, im_h = im.size
    mat_path = im_path.replace('.jpg', '.mat').replace('images', 'density_maps')
    points = loadmat(mat_path)['annPoints'].astype(np.float32)
    idx_mask = (points[:, 0] >= 0) * (points[:, 0] <= im_w) * (points[:, 1] >= 0) * (points[:, 1] <= im_h)
    points = points[idx_mask]
    im_h, im_w, rr = cal_new_size(im_h, im_w, min_size, max_size)
    im = np.array(im)
    #if rr != 1.0:
    im = cv2.resize(np.array(im), (im_w, im_h), cv2.INTER_CUBIC)
    points = points * rr
    return Image.fromarray(im), points


def parse_args():
    """ Parses arguments containing the original and processed data directories
    """
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--origin-dir', default='../Datasets/UCF-CC-50/folds/fold_1',
                        help='original data directory')
    parser.add_argument('--data-dir', default='../Datasets/UCF-CC-50Preprocessed/folds/fold_1',
                        help='processed data directory')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    save_dir = args.data_dir
    min_size = 256
    max_size = 2048

    sub_dir = args.origin_dir
    sub_save_dir = save_dir
    if not os.path.exists(sub_save_dir):
        os.makedirs(sub_save_dir)
    im_list = glob(os.path.join(sub_dir, '*jpg'))
    print(sub_dir)
    for im_path in im_list:
        name = os.path.basename(im_path)
        print(name)
        im, points = generate_data(im_path)
        im_save_path = os.path.join(sub_save_dir, name)
        im.save(im_save_path)
        gd_save_path = im_save_path.replace('jpg', 'npy')
        np.save(gd_save_path, points)
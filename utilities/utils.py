# Taken from ConNet Repository

import os
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

import xlsxwriter


def to_var(x, use_gpu, requires_grad=False):
    """ Toggles the use of cuda of a Tensor variable

        :param torch.Tensor x: Tensor variable to toggle the CUDA of
        :param boolean use_gpu: whether the use of GPU is permitted
        :param boolean requires_grad: whether gradients must be completed

        :returns: Modified Tensor variable

        :rtype: torch.Tensor
    """
    if torch.cuda.is_available() and use_gpu:
        x = x.cuda()
    x.requires_grad = requires_grad
    return x


def mkdir(directory):
    """ Creates the directory if not yet existing

        :param string directory: directory to be created
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_print(path, text):
    """ Displays text in console and saves in text file

        :param string path: path to text file
        :param string text: text to display and save
    """
    file = open(path, 'a')
    file.write(text + '\n')
    file.close()
    print(text)

def write_to_file(path, text):
    """ Saves text in text file

        :param string path: path to text file
        :param string text: text to display and save
    """
    file = open(path, 'a')
    file.write(text + '\n')
    file.close()

def save_plots(file_path, output, labels, ids, save_label=False):
    """ Saves the density maps as images

        :param string file_path: path to save the images
        :param torch.Tesnor output: density map outputted by the model
        :param torch.Tensor labels: groundtruth density map
        :param list ids: list of the file names of the dataset images
        :param boolean save_label: whether the labels should be saved as images
    """

    # output folder
    dm_file_path = os.path.join(file_path, 'density maps')
    mkdir(dm_file_path)

    # file paths
    img_file_path = os.path.join("C:/Users/lande/Desktop/THS-ST2/Datasets/ShanghaiTechAPreprocessed/test", '%s')
    file_path = os.path.join(file_path , '%s')
    dm_file_path = os.path.join(dm_file_path, '%s')
    img_dest_path = dm_file_path.replace("density maps", "images")

    # save the original image, GT density map, and generated density maps into a folder
    print('Saving density maps and creating summary sheet...')

    workbook = xlsxwriter.Workbook(file_path.split("/")[-1] + '.xlsx')
    print("Saving into sheet:", file_path.split("/")[-1] + '.xlsx')

    worksheet = workbook.add_worksheet()
    row = 0
    column = 0
    worksheet.write(row, column, "Ground Truth")
    worksheet.write(row, column + 1, "Predicted Count")
    worksheet.write(row, column + 2, "Absolute Difference")
    worksheet.write(row, column + 3, "Error Rate")

    for i in range(0, len(ids)):
        # # save image
        # img_file_name = img_file_path % (ids[i])
        # plt.imsave(img_file_name)

        # save density map outputted by the model
        file_name = dm_file_path % (ids[i])
        # o = output[i].cpu().detach().numpy()
        o = output[i].cpu().detach().numpy()
        et_count = np.sum(o)
        o = o.squeeze()
        plt.imsave(file_name, o)

        # prepare other file names
        file_name2 = file_path % (ids[i])
        file_name3 = dm_file_path % ("[gt] {}".format(ids[i]))

        # save the ground-truth density map
        l = labels[i].cpu().detach().numpy()
        gt_count = np.sum(l)
        l = l.squeeze()

        if save_label:
            plt.imsave(file_name3, l)

        img = plt.imread(img_file_path % ids[i])
        plt.subplot(1, 3, 1)
        plt.imshow(img)

        # plot the two density maps in the same image
        plt.subplot(1, 3, 2)
        plt.imshow(l)
        text = plt.text(0, 0, 'actual: {} ({})\npredicted: {} ({})\n\n'.format(round(gt_count), str(gt_count), round(et_count), str(et_count)))
        
        plt.subplot(1, 3, 3)
        plt.imshow(o)
        plt.savefig(file_name2)

        text.set_visible(False)

        # input into summary sheet
        worksheet.write(row + i + 1, column, gt_count)
        worksheet.write(row + i + 1, column + 1, et_count)
        worksheet.write(row + i + 1, column + 2, abs(et_count - gt_count))
        worksheet.write(row + i + 1, column + 3, abs(et_count - gt_count)/gt_count*100)
        
    workbook.close()

def get_amp_gt_by_value(target, threshold=1e-5):
    """ Creates the attention map groundtruth used by MARUNet

        :param torch.Tensor target: groundtruth density map
        :param float threshold: threshold value used for generating the attention map
    """
    seg_map = (target>threshold).float().cuda()
    return seg_map
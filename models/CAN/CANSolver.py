import glob
import math
import os

from models.CAN.CAN import CANNet
from models.CAN.CANUtils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
import models.CAN.CANDataset as CANDataset
import time
from datetime import date
import PIL.Image as Image
import h5py

from sklearn.metrics import mean_squared_error,mean_absolute_error

from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt

class AverageMeter(object):
    def __init__(self):
        """ Initializes an AverageMeter object
        """
        self.reset()

    def reset(self):
        """ Resets the values of the AverageMeter object
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Updates the values of the AverageMeter object
        
            :param int val: value of val
            :param int n: value of n
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CANSolver(object):
    def __init__(self, config, paths):
        """
        Initializes a CAN Solver object

        Arguments:
            config {Object} -- configurations of the model
            paths {Object} -- paths to the resources used by the model  
        """
        self.config = config
        self.paths = paths
        self.lr = self.config.lr
        self.workers = 4
        self.start_epoch = 0
        self.print_freq = 30
        self.data = ''

        self.build_model()

    def build_model(self):
        """ Instantiates the model, loss criterion, and optimizer
        """
        self.model = CANNet()
        self.criterion = nn.MSELoss(size_average=False)
#         self.criterion = nn.L1Loss()
#         if self.config.dataset == "Shanghaitech-A" or self.config.dataset == "UCFCC50":
        if self.config.dataset == "":
            self.optimizer = torch.optim.SGD(params=self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr = self.config.lr, weight_decay=self.config.weight_decay)
        
        if self.config.dataset == 'Shanghaitech-A':
            self.data = self.paths.shanghaitech_a
        elif self.config.dataset == 'Shanghaitech-B':
            self.data = self.paths.shanghaitech_b
        elif self.config.dataset == 'UCFCC50':
            self.data = self.paths.ucf_cc_50
        elif self.config.dataset == 'UCFQNRF':
            self.data = self.paths.ucf_qnrf
        
        # enable GPU use if available and set to True by user
        if torch.cuda.is_available() and self.config.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

        # print network
        print(self.model)
        print(self.optimizer)
    
    def start(self, config):
        """ Starts model training
        
            :param Object config: configurations of the model
        """
        if self.config.dataset == 'UCFCC50':
            save_folder_name = str(config.model) + ' ' + config.dataset + '_fold' + str(self.config.cc50_val) + ' ' + str(date.today().strftime("%d-%m-%Y") + ' ' + str(time.strftime("%H_%M_%S", time.localtime())))
        else:
            save_folder_name = str(config.model) + ' ' + config.dataset + ' ' + str(date.today().strftime("%d-%m-%Y") + ' ' + str(time.strftime("%H_%M_%S", time.localtime())))
        self.weights_save_path = os.path.join('./weights', save_folder_name)

        if self.config.mode == "Train":
            start_time = time.time()

            best_prec1 = 1e6
            best_rmse = 1e6
            best_epoch = 0

            try:
                os.makedirs(self.weights_save_path)
                print("Directory successfully created")
            except:
                print("Directory already exists")
            self.log_path = os.path.join(self.weights_save_path, 'train_log.txt')
            f = open(self.log_path, "w")
            f.close()

            self.sched = 0
            
            for e in range(self.start_epoch, self.config.num_epochs):

                self.train(self.model, self.criterion, self.optimizer, e, f, self.config)
                prec1, rmse = self.validate(self.model, self.criterion, self.config)

                is_best = prec1 < best_prec1

                if(best_prec1 > min(prec1, best_prec1)):
                    best_epoch = e

                best_prec1 = min(prec1, best_prec1)
                best_rmse = min(rmse, best_rmse)

                print(' * best MAE {mae:.9f}, best RMSE {rmse:.9f} '
                    .format(mae=best_prec1, rmse=best_rmse))

#                 print(' * best RMSE {rmse:.9f} '
#                     .format(rmse=best_rmse))

                save_checkpoint({
                    'epoch': e + 1,
                    'arch': self.paths.pretrained_model,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : self.optimizer.state_dict(),
                }, is_best, self.config, self.weights_save_path)

                print(' * best epoch: %d' % (best_epoch))
                print('current time: %s' % (time.time() - start_time))

                f = open(self.log_path, "a")
                f.write('current MAE: {mae:.9f}, RMSE: {rmse:.9f}\n'.format(mae=prec1, rmse=rmse))
#                 f.write('current RMSE: {rmse:.9f}\n'.format(rmse=rmse))
                f.write(' * best MAE: {mae:.9f}, best RMSE: {rmse:.9f}\n'.format(mae=best_prec1, rmse=best_rmse))
#                 f.write(' * best RMSE: {rmse:.9f}\n'.format(rmse=best_rmse))
                f.write(' * best epoch: %d\n' % (best_epoch))
                f.write('current time: %s\n' % (time.time() - start_time))
                f.close()

        elif self.config.mode == "Test":
            # load using .pth.tar file
            if ".pth.tar" in self.config.weights:
                checkpoint = torch.load(os.path.join('./weights', self.config.weights))
                self.model.load_state_dict(checkpoint['state_dict'])

            # load using .pth file   
            else:
                self.model.load_state_dict(torch.load(os.path.join('./weights', self.config.weights)), strict=False)

            self.test()
            
    
    def train(self, model, criterion, optimizer, epoch, f, config):
        """ Performs model training
        
            :param Object model: model to be used
            :param Object criterion: criterion to be used
            :param Object optimizer: optimizer to be used
            :param int epoch: starting epoch
            :param File f: file where the training details are saved
            :param Object config: configurations used for training
        """
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        train_loader = torch.utils.data.DataLoader(
            CANDataset.listDataset(config, self.data,
                        shuffle=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                    ]),
                        train=True,
                        seen=model.seen,
                        batch_size=self.config.batch_size,
                        num_workers=self.workers),
            batch_size=self.config.batch_size)

        print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), self.lr))

        model.train()
        end = time.time()
        
        num_sched = len(self.config.learning_sched)
        if num_sched != 0 and self.sched < num_sched:
            if (epoch + 1) in self.config.learning_sched:
                self.lr *= 10
                print('Learning rate increased to', self.lr)

#                 if self.config.dataset == "Shanghaitech-A" or self.config.dataset == "UCFCC50":
                if self.config.dataset == '':
                    optimizer = torch.optim.SGD(params=model.parameters(), lr=self.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
                else:
                    optimizer = torch.optim.Adam(params=model.parameters(), lr = self.lr, weight_decay=self.config.weight_decay)

                self.sched += 1

        for i,(img, target) in enumerate(tqdm(train_loader)):
            data_time.update(time.time() - end)

            img = img.cuda()
            img = Variable(img)
            output = model(img)[:,0,:,:]

            target = target.type(torch.FloatTensor).cuda()
            target = Variable(target)

            loss = criterion(output, target)

            losses.update(loss.item(), img.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if math.isnan(losses.val) or math.isnan(losses.avg):
                        print("NaN loss value detected, terminating...")
                        f.write("NaN loss value detected, terminating...")
                        quit()
        
        print('Epoch: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.9f} ({loss.avg:.9f})\t'
            .format(
            epoch, self.config.num_epochs, batch_time=batch_time,
            data_time=data_time, loss=losses))
        
        f = open(self.log_path, "a")
        f.write('Epoch: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.9f} ({loss.avg:.9f})\t\n'
            .format(
            epoch, self.config.num_epochs, batch_time=batch_time,
            data_time=data_time, loss=losses))
        f.close()        
    
    def validate(self, model, criterion, config):
        """ Performs model validation
        
            :param Object model: model to be evaluated
            :param Object criterion: criterion to be used
            :param Object config: configurations used for model validation

            :returns:
                - (:py:class:`double`) - resulting MAE of the model evaluation
                - (:py:class:`double`) - resulting RMSE of the model evaluation
        """
        test_loader = torch.utils.data.DataLoader(
        CANDataset.listDataset(config, self.data,
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ]),  train=False),
        batch_size=1)

        model.eval()

        mae = 0
        rmse = 0

        for i,(img, target) in enumerate(test_loader):
            h,w = img.shape[2:4]
            h_d = int(h/2)
            w_d = int(w/2)
            img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
            img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
            img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
            img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
            density_1 = model(img_1).data.cpu().numpy()
            density_2 = model(img_2).data.cpu().numpy()
            density_3 = model(img_3).data.cpu().numpy()
            density_4 = model(img_4).data.cpu().numpy()

            pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()

            mae += abs(pred_sum-target.sum())
            rmse += (pred_sum-target.sum())**2

        mae = mae/len(test_loader)
        rmse = rmse/len(test_loader)
        rmse = math.sqrt(rmse)

        print('MAE {mae:.9f}, RMSE {rmse:.9f} '
                .format(mae=mae, rmse=rmse))

#         print(' * RMSE {rmse:.9f} '
#                 .format(rmse=rmse))

        return mae, rmse
    
    def test(self):
        """ Performs model testing
        """
        self.tests_save_path = os.path.join('./tests', self.config.weights.split("/")[1])
        try:
            print("Creating test save path directory...")
            os.makedirs(self.tests_save_path)
            print("Directory successfully created")
        except:
            print("Directory already exists")
        self.testlog_path = os.path.join(self.tests_save_path, 'test_log.txt')

        transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ])
        if self.config.dataset == 'UCFCC50':
            img_folder = os.path.join(self.data, 'fold_5', 'images')
        else:
            img_folder = os.path.join(self.data, 'test', 'images')
        img_paths = []

        for img_path in glob.glob(os.path.join(img_folder, '*.jpg')):
            img_paths.append(img_path)
        
        self.model.eval()
        start_time = time.time()

        pred= []
        gt = []

        images = []
        density_maps_generated = []
        density_maps_gt = []

        for i in tqdm(range(len(img_paths))):
            ### added images list
            images.append(Image.open(img_paths[i]).convert('RGB'))

            img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
            img = img.unsqueeze(0)
            h,w = img.shape[2:4]
            h_d = int(h/2)
            w_d = int(w/2)
            img_1 = Variable(img[:,:,:h_d,:w_d].cuda())
            img_2 = Variable(img[:,:,:h_d,w_d:].cuda())
            img_3 = Variable(img[:,:,h_d:,:w_d].cuda())
            img_4 = Variable(img[:,:,h_d:,w_d:].cuda())
            density_1 = self.model(img_1).data.cpu().numpy()
            density_2 = self.model(img_2).data.cpu().numpy()
            density_3 = self.model(img_3).data.cpu().numpy()
            density_4 = self.model(img_4).data.cpu().numpy()

            ### added generated density map list
            img_density = self.model(img).data.cpu().numpy()
            density_map = img_density.squeeze(0).squeeze(0)
            density_maps_generated.append(density_map)

            pure_name = os.path.splitext(os.path.basename(img_paths[i]))[0]
            gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','density_maps'),'r')
            groundtruth = np.asarray(gt_file['density'])

            ### added groundtruth density map list
            density_maps_gt.append(groundtruth)

            pred_sum = density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
            pred.append(pred_sum)
            gt.append(np.sum(groundtruth))

        mae = mean_absolute_error(pred,gt)
        rmse = np.sqrt(mean_squared_error(pred,gt))

        print(self.config.weights.split("/")[1] + ":")
        print('MAE: ',mae)
        print('RMSE: ',rmse)

        f = open(self.testlog_path, "w")
        f.write(self.config.weights.split("/")[1] + "\n")
        f.write('MAE: {mae:.9f}\n'.format(mae=mae))
        f.write('RMSE: {rmse:.9f}\n'.format(rmse=rmse))
        f.close()

        # save the original image, GT density map, and generated density maps into a folder
        print('Saving density maps...')
        for n in tqdm(range(len(images))):
            f = plt.figure()
            f.add_subplot(1,3,1).set_title('Original Image')
            plt.imshow(images[n])
            text = plt.text(0, 0, 'actual: {} ({})\npredicted: {} ({})\n\n'.format(round(gt[n]), str(gt[n]), round(pred[n]), str(pred[n])))
            f.add_subplot(1,3,2).set_title('Ground Truth Density Map')
            plt.imshow(density_maps_gt[n])
            f.add_subplot(1,3,3).set_title('Generated Density Map')
            f.set_size_inches(20, 5)
            plt.imshow(density_maps_generated[n])

            # for local saving
#             filename = os.path.join(self.tests_save_path, img_paths[n].split('\\')[-1])

            # for jupyterhub saving
            filename = os.path.join(self.tests_save_path, img_paths[n].split('/')[-1])

            f.savefig(filename)
            text.set_visible(False)
            plt.close()

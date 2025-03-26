import glob
import math
import os

from models.CAN.CAN import CANNet
from compression.CAN_teacher import CANNet as CAN_teacher
from compression.CAN_student import CANNet as CAN_student
from models.CAN.CANUtils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

import numpy as np
import models.CAN.CANDataset as CANDataset
import models.CAN.CANDatasetSKT as CANDatasetSKT
import time
from datetime import date
import PIL.Image as Image
import h5py

from sklearn.metrics import mean_squared_error,mean_absolute_error

from tqdm import tqdm as tqdm
from matplotlib import pyplot as plt

from compression.skt_utils import cal_para, cosine_similarity, scale_process_CAN, cal_dense_fsp

import xlsxwriter

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


class CANSolverSKT(object):
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
        """ Instantiates the student and teacher models, loss criterion, and optimizer
        """
        self.teacher_model = CAN_teacher()
        print('(Teacher Model) ', end='')
        print_num_params(self.teacher_model)  # include 1x1 conv transform parameters
        print('(Student Model) ', end='')
        self.student_model = CAN_student(ratio=8)
        print_num_params(self.student_model)  # include 1x1 conv transform parameters

        self.teacher_model.regist_hook()  # use hook to get teacher's features

        self.criterion = nn.MSELoss(size_average=False)
        
        if self.config.dataset == "":
            self.optimizer = torch.optim.SGD(params=self.student_model.parameters(), lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(params=self.student_model.parameters(), lr = self.config.lr, weight_decay=self.config.weight_decay)
        
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
            self.teacher_model = self.teacher_model.cuda()
            self.student_model = self.student_model.cuda()
            self.criterion.cuda()

        # print network
        print("TEACHER MODEL:")
        print(self.teacher_model)
        print("STUDENT MODEL:")
        print(self.student_model)
        print(self.optimizer)
    
    def start(self, config):
        """ Prunes the model and starts model training
        
            :param Object config: configurations of the model
        """
        if self.config.dataset == 'UCFCC50':
            save_folder_name = 'SKT-' + str(config.model) + ' ' + config.dataset + '_fold' + str(self.config.cc50_val) + ' ' + str(date.today().strftime("%d-%m-%Y") + ' ' + str(time.strftime("%H_%M_%S", time.localtime())))
        else:
            save_folder_name = 'SKT-' + str(config.model) + ' ' + config.dataset + ' ' + str(date.today().strftime("%d-%m-%Y") + ' ' + str(time.strftime("%H_%M_%S", time.localtime())))
        self.weights_save_path = os.path.join('./weights', save_folder_name)

        if self.config.mode == "Train":
            
            if self.config.weights is not None:
                # load using .pth.tar file
                if ".pth.tar" in self.config.weights:
                    checkpoint = torch.load(os.path.join('./weights', self.config.weights))
                    self.student_model.load_state_dict(checkpoint['state_dict'])

                # load using .pth file   
                else:
                    self.student_model.load_state_dict(torch.load(os.path.join('./weights', self.config.weights)), strict=False)
                    
                print("Successfully loaded weights from", self.config.weights)
                
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

                self.train(self.teacher_model, self.student_model, self.criterion, self.optimizer, e, f, self.config)
                prec1, rmse = self.validate(self.student_model, self.criterion, self.config)

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
                    'state_dict': self.student_model.state_dict(),
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
                self.student_model.load_state_dict(checkpoint['state_dict'])

            # load using .pth file   
            else:
                self.student_model.load_state_dict(torch.load(os.path.join('./weights', self.config.weights)), strict=False)

            self.test()
            
    
    def train(self, teacher_model, student_model, criterion, optimizer, epoch, f, config):
        """ Performs model training
        
            :param Object model: model to be used
            :param Object criterion: criterion to be used
            :param Object optimizer: optimizer to be used
            :param int epoch: starting epoch
            :param File f: file where the training details are saved
            :param Object config: configurations used for training
        """
        losses_h = AverageMeter()
        losses_s = AverageMeter()
        losses_fsp = AverageMeter()
        losses_cos = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        train_loader = torch.utils.data.DataLoader(
            CANDatasetSKT.listDataset(config, self.data,
                        transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                    ]),
                        train=True,
                        seen=student_model.seen,
                        batch_size=self.config.batch_size,
                        num_workers=self.workers),
            batch_size=self.config.batch_size)

        print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), self.lr))

        teacher_model.eval()
        student_model.train()
        end = time.time()
        
        num_sched = len(self.config.learning_sched)
        if num_sched != 0 and self.sched < num_sched:
            if (epoch + 1) in self.config.learning_sched:
                self.lr *= 10
                print('Learning rate increased to', self.lr)

#                 if self.config.dataset == "Shanghaitech-A" or self.config.dataset == "UCFCC50":
                if self.config.dataset == '':
                    optimizer = torch.optim.SGD(params=student_model.parameters(), lr=self.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)
                else:
                    optimizer = torch.optim.Adam(params=student_model.parameters(), lr = self.lr, weight_decay=self.config.weight_decay)

                self.sched += 1

        for i,(img, target) in enumerate(tqdm(train_loader)):
            data_time.update(time.time() - end)

            img = img.cuda()
            img = Variable(img)
            # output = model(img)[:,0,:,:]
            target = target.type(torch.FloatTensor).cuda()
            target = Variable(target)

            with torch.no_grad():
                teacher_output = teacher_model(img)
                teacher_model.features.append(teacher_output)
                teacher_fsp_features = [scale_process_CAN(teacher_model.features)]
                teacher_fsp = cal_dense_fsp(teacher_fsp_features)
            
            student_features = student_model(img)
            student_output = student_features[-1]
            student_fsp_features = [scale_process_CAN(student_features)]
            student_fsp = cal_dense_fsp(student_fsp_features)

            loss_h = criterion(student_output, target)
            loss_s = criterion(student_output, teacher_output)

            loss_fsp = torch.tensor([0.], dtype=torch.float).cuda()
            if config.lamb_fsp:
                loss_f = []
                assert len(teacher_fsp) == len(student_fsp)
                for t in range(len(teacher_fsp)):
                    loss_f.append(criterion(teacher_fsp[t], student_fsp[t]))
                loss_fsp = sum(loss_f) * config.lamb_fsp

            loss_cos = torch.tensor([0.], dtype=torch.float).cuda()
            if config.lamb_cos:
                loss_c = []
                for t in range(len(student_features) - 1):
                    loss_c.append(cosine_similarity(student_features[t], teacher_model.features[t]))
                loss_cos = sum(loss_c) * config.lamb_cos

            loss = loss_h + loss_s + loss_fsp + loss_cos

            losses_h.update(loss_h.item(), img.size(0))
            losses_s.update(loss_s.item(), img.size(0))
            losses_fsp.update(loss_fsp.item(), img.size(0))
            losses_cos.update(loss_cos.item(), img.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if math.isnan(losses_h.val) or math.isnan(losses_h.avg) or \
                math.isnan(losses_s.val) or math.isnan(losses_s.avg) or \
                math.isnan(losses_fsp.val) or math.isnan(losses_fsp.avg) or \
                math.isnan(losses_cos.val) or math.isnan(losses_cos.avg):
                        print("NaN loss value detected, terminating...")
                        f.write("NaN loss value detected, terminating...")
                        quit()
        
        print('Epoch: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss_h {loss_h.avg:.4f}  '
            'Loss_s {loss_s.avg:.4f}  '
            'Loss_fsp {loss_fsp.avg:.4f}  '
            'Loss_cos {loss_kl.avg:.4f}  '
            .format(
            epoch, self.config.num_epochs, batch_time=batch_time,
            data_time=data_time, loss_h=losses_h, loss_s=losses_s,
            loss_fsp=losses_fsp, loss_kl=losses_cos))
        
        f = open(self.log_path, "a")
        f.write('Epoch: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss_h {loss_h.avg:.4f}  '
            'Loss_s {loss_s.avg:.4f}  '
            'Loss_fsp {loss_fsp.avg:.4f}  '
            'Loss_cos {loss_kl.avg:.4f}\t\n'
            .format(
            epoch, self.config.num_epochs, batch_time=batch_time,
            data_time=data_time, loss_h=losses_h, loss_s=losses_s,
            loss_fsp=losses_fsp, loss_kl=losses_cos))
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
        # initially validated using CANDataset class not CANDatasetSKT
        CANDatasetSKT.listDataset(config, self.data,
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
        
        self.student_model.eval()
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
            density_1 = self.student_model(img_1).data.cpu().numpy()
            density_2 = self.student_model(img_2).data.cpu().numpy()
            density_3 = self.student_model(img_3).data.cpu().numpy()
            density_4 = self.student_model(img_4).data.cpu().numpy()

            ### added generated density map list
            img_density = self.student_model(img).data.cpu().numpy()
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
        f.write('total time: %s\n' % (time.time() - start_time))
        f.close()

        # save the original image, GT density map, and generated density maps into a folder
        print('Saving density maps and creating summary sheet...')

        workbook = xlsxwriter.Workbook(self.config.weights.split("/")[1] + '.xlsx')
        print("Saving into sheet:", self.config.weights.split("/")[1] + '.xlsx')

        worksheet = workbook.add_worksheet()
        row = 0
        column = 0
        worksheet.write(row, column, "Ground Truth")
        worksheet.write(row, column + 1, "Predicted Count")
        worksheet.write(row, column + 2, "Absolute Difference")
        worksheet.write(row, column + 3, "Error Rate")

        for n in tqdm(range(len(images))):

            # save density map
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

            # input into summary sheet
            worksheet.write(row + n + 1, column, gt[n])
            worksheet.write(row + n + 1, column + 1, pred[n])
            worksheet.write(row + n + 1, column + 2, abs(pred[n] - gt[n]))
            worksheet.write(row + n + 1, column + 3, abs(pred[n] - gt[n])/gt[n]*100)
            
        workbook.close()

def print_num_params(model):
    """ Prints the structure of the network and the total number of parameters

            :param Object model: the model to be used
        """
    num_params = 0
    for name, param in model.named_parameters():
        if 'transform' in name:
            continue
        num_params += param.data.count_nonzero()
    print('The number of parameters: {}'.format(num_params))
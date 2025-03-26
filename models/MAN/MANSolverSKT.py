from models.MAN.MANUtils import Trainer
from models.MAN.MANUtils import Save_Handle, AverageMeter
import os
import sys
import time
import torch
from torch import optim
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import logging
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from models.MAN.MAN import MAN
from compression.MAN_teacher import MAN as MANteacher
from compression.MAN_student import MAN as MANstudent
from models.MAN.MANDataset import Crowd
from models.MAN.MANLoss import Bay_Loss
from models.MAN.MANLoss import Post_Prob
from math import ceil

from compression.skt_utils import cal_para, cosine_similarity, scale_process, cal_dense_fsp
from models.MAN.MANUtils import print_num_params

import gc

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

__all__ = ['vgg19_trans']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

def init_MAN_teacher():
    """ Initializes the MAN teacher model
    
        :returns: Initialized model

        :rtype: Object
    """
    # Use the VGG 19-layer model (configuration "E") pre-trained on ImageNet
    model = MANteacher(make_layers(cfg['E']))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
  
def init_MAN_student(ratio):
    """ Initializes the MAN student model
    
        :returns: Initialized model

        :rtype: Object
    """
    # Use the VGG 19-layer model (configuration "E") pre-trained on ImageNet
    model = MANstudent(make_layers(cfg['E']), ratio)
    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model

def make_layers(cfg, batch_norm=False):
    """ Creates the layers of the model
    
        :param list cfg: number of channels per layer of the model
        :param boolean batch_norm: whether batch normalization is to be implemented

        :returns: Sequential container storing the layers of the model

        :rtype: nn.Sequential
    """
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def train_collate(batch):
    """ Collates the relevant details of the batch of input images for model training
    
        :param list batch: batch of input images

        :returns:
            - (:py:class:`torch.Tensor`) - tensor representation of the list of input images
            - (:py:class:`list`) - list of tensor representations of the ground truth density maps
            - (:py:class:`list`) - list of tensor representations of the generated density maps
            - (:py:class:`torch.FloatTensor`) - tensor representation of the list of minimum dimensions of the input images
    """
    transposed_batch = list(zip(*batch))
    images = torch.stack(transposed_batch[0], 0)
    points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
    targets = transposed_batch[2]
    st_sizes = torch.FloatTensor(transposed_batch[3])
    return images, points, targets, st_sizes
    
class MANSolverSKT(Trainer):
    def setup(self):
        """ Initializes the datasets, model, loss, and optimizer
        """
        args = self.args
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            # for code conciseness, we release the single gpu version
            assert self.device_count == 1
            logging.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.downsample_ratio = args.downsample_ratio
        
        #ADDED CODE FOR WARMUP LEARNING
        
        self.learning_sched = args.learning_sched
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.lamb_fsp = args.lamb_fsp
        self.lamb_cos = args.lamb_cos

        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.dataset_name, args.cc_50_val, args.cc_50_test, args.is_gray, args.augment_contrast, args.augment_contrast_factor, args.augment_save_location, args.augment_save, x) for x in ['train', 'val']}
            
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'val']}

        self.teacher_model = init_MAN_teacher()
        self.student_model = init_MAN_student(ratio=4)
        self.teacher_model.to(self.device)
        self.student_model.to(self.device)

        self.optimizer = optim.Adam(self.student_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        
        if args.teacher_ckpt:
            if os.path.isfile(args.teacher_ckpt):
                print("=> loading checkpoint '{}'".format(args.teacher_ckpt))
                checkpoint = torch.load(args.teacher_ckpt)
                self.teacher_model.load_state_dict(checkpoint['model_state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.teacher_ckpt, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.teacher_ckpt))

        if args.student_ckpt:
            if os.path.isfile(args.student_ckpt):
                print("=> loading checkpoint '{}'".format(args.student_ckpt))
                checkpoint = torch.load(args.student_ckpt)
                args.start_epoch = checkpoint['epoch']
                self.best_mae = checkpoint['best_mae']
                self.best_mse = checkpoint['best_mse']
                self.student_model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = args.start_epoch + 1
                print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.student_ckpt, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(args.student_ckpt))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device).cuda()
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_all = args.save_all
        self.best_count = 0

        print_num_params(self.student_model, self.student_model)
        #cal_para(self.student_model)
        self.teacher_model.regist_hook()
        self.teacher_model = self.teacher_model.cuda()
        self.student_model = self.student_model.cuda()

    def train(self):
        """ Performs model training
        """
        args = self.args
        
        #ADDED VARIABLE INITIALIZATION FOR WARMUP LEARNING
        self.sched = 0
        
        for epoch in range(self.start_epoch, args.max_epoch):
            self.epoch = epoch
            self.train_epoch()
            
            #ADDED CODE FOR WARMUP LEARNING   
                
            num_sched = len(self.learning_sched)
            if num_sched != 0 and self.sched < num_sched:
                if (self.epoch + 1) in self.learning_sched:
                    self.lr = self.lr * 10
                    print("Learning rate decreased to ", self.lr)
                    self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                    self.sched += 1
            
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                #mae_prec1, mse_prec1 = self.val_epoch()
                self.val_epoch()

    def train_epoch(self):
        """ Performs a single epoch of model training
        """
        losses_h = AverageMeter()
        losses_s = AverageMeter()
        losses_fsp = AverageMeter()
        losses_cos = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        epoch_start = time.time()
        
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()

        self.teacher_model.eval()
        self.student_model.train()
        
        end = time.time()

        #ADDED CODE FOR WARMUP LEARNING   
                
        num_sched = len(self.learning_sched)
        if num_sched != 0 and self.sched < num_sched:
            if (self.epoch + 1) in self.learning_sched:
                self.lr *= 10
                print("Learning rate increased to ", self.lr)
                self.optimizer = optim.Adam(self.student_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                self.sched += 1
                        
        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.no_grad():
                teacher_features = self.teacher_model(inputs)
                self.teacher_model.features.append(teacher_features[0])
                teacher_fsp_features = [scale_process(self.teacher_model.features)]
                teacher_fsp = cal_dense_fsp(teacher_fsp_features)

            student_features = self.student_model(inputs)
            student_output = student_features[-1]
            student_fsp_features = [scale_process(self.student_model.features)]
            student_fsp = cal_dense_fsp(student_fsp_features)

            prob_list = self.post_prob(points, st_sizes)

            loss_h = self.criterion(prob_list, targets, student_features[0])
            loss_s = self.criterion(prob_list, teacher_features[0].tolist()[0][0][0], student_features[0])

            loss_fsp = torch.tensor([0.], dtype=torch.float).cuda()

            if self.lamb_fsp:
                loss_f = []
                assert len(teacher_fsp) == len(student_fsp)
                for t in range(len(teacher_fsp)):
                    loss_f.append(self.criterion(student_fsp[t], teacher_fsp[t]))
                loss_fsp = sum(loss_f) * self.lamb_fsp

            loss_cos = torch.tensor([0.], dtype=torch.float).cuda()
            if self.lamb_cos:
                loss_c = []
                for t in range(len(student_features) - 1):
                    loss_c.append(cosine_similarity(student_features[t], self.teacher_model.features[t]))
                loss_cos = sum(loss_c) * self.lamb_cos

            loss = loss_h + loss_s + loss_fsp + loss_cos

            losses_h.update(loss_h.item(), inputs.size(0))
            losses_s.update(loss_s.item(), inputs.size(0))
            losses_fsp.update(loss_fsp.item(), inputs.size(0))
            losses_cos.update(loss_cos.item(), inputs.size(0))
                
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            N = inputs.size(0)
            pre_count = torch.sum(student_features[0].view(N, -1), dim=1).detach().cpu().numpy()
            res = pre_count - gd_count
            epoch_loss.update(loss.item(), N)
            epoch_mse.update(np.mean(res * res), N)
            epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                 .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                         time.time()-epoch_start))
        
        model_state_dic = self.student_model.state_dict()
        
        print('Epoch: [{0}]\t'
                  'Time {batch_time.avg:.3f}  '
                  'Data {data_time.avg:.3f}  '
                  'Loss_h {loss_h.avg:.4f}  '
                  'Loss_s {loss_s.avg:.4f}  '
                  'Loss_fsp {loss_fsp.avg:.4f}  '
                  'Loss_cos {loss_kl.avg:.4f}  '
                .format(
                self.epoch, batch_time=batch_time,
                data_time=data_time, loss_h=losses_h, loss_s=losses_s,
                loss_fsp=losses_fsp, loss_kl=losses_cos))
        
        model_state_dic_teacher = self.teacher_model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_teacher_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic_teacher
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models
        
        model_state_dic_student = self.student_model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_student_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic_student,
            'best_mae': self.best_mae,
            'best_mse': self.best_mse
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models
        
        val_epoch_mse = np.sqrt(epoch_mse.get_avg())
        val_epoch_mae = epoch_mae.get_avg()

        print("mae {:.2f} mse {:.2f}".format(val_epoch_mae, val_epoch_mse))
         
    def val_epoch(self):
        """ Performs model validation
        """
        epoch_start = time.time()
        epoch_res = []

        #self.student_model.eval()

        # Iterate over data.
        for inputs, count, name in self.dataloaders['val']:
            inputs = inputs.to(self.device)
            # inputs are images with different sizes
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            if h >= 3584 or w >= 3584:
                h_stride = int(ceil(1.0 * h / 3584))
                w_stride = int(ceil(1.0 * w / 3584))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.no_grad():
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.student_model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.no_grad():
                    outputs = self.student_model(inputs)[0]
                    # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.student_model.state_dict()
        logging.info("best mse {:.2f} mae {:.2f}".format(self.best_mse, self.best_mae))
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.epoch)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
        
    def test(self, args):
        """ Performs model testing
        
            :param Object args: arguments used by the model
        """
        args = args

        datasets = Crowd(os.path.join(args.data_dir, 'test'),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.dataset_name, args.cc_50_val, args.cc_50_test, args.is_gray, args.augment_contrast, args.augment_contrast_factor, args.augment_save_location, augment_save, method='test')

        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                                 num_workers=8, pin_memory=False)

        device = torch.device('cuda')
        model = init_MAN_student(ratio=4)
        model.to(device)
        model.eval()

        model.load_state_dict(torch.load(args.best_model_path, device))
        epoch_minus = []
        
        epoch_start = time.time()
        
        for inputs, count, name in dataloader:
            print(name)
            inputs = inputs.to(device)
            b, c, h, w = inputs.shape
            h, w = int(h), int(w)
            assert b == 1, 'the batch size should equal to 1 in validation mode'
            input_list = []
            if h >= 3584 or w >= 3584:
                h_stride = int(ceil(1.0 * h / 3584))
                w_stride = int(ceil(1.0 * w / 3584))
                h_step = h // h_stride
                w_step = w // w_stride
                for i in range(h_stride):
                    for j in range(w_stride):
                        h_start = i * h_step
                        if i != h_stride - 1:
                            h_end = (i + 1) * h_step
                        else:
                            h_end = h
                        w_start = j * w_step
                        if j != w_stride - 1:
                            w_end = (j + 1) * w_step
                        else:
                            w_end = w
                        input_list.append(inputs[:, :, h_start:h_end, w_start:w_end])
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_minus.append(res)
            else:
                with torch.no_grad():
                    outputs = model(inputs)[0]
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_minus.append(res)
            print(res)

        epoch_minus = np.array(epoch_minus)
        mse = np.sqrt(np.mean(np.square(epoch_minus)))
        mae = np.mean(np.abs(epoch_minus))
        log_str = 'mae {}, mse {}'.format(mae, mse)
        print(log_str)
        print(time.time() - epoch_start)
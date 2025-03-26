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
from models.MAN.MANDataset import Crowd
from models.MAN.MANLoss import Bay_Loss
from models.MAN.MANLoss import Post_Prob
from math import ceil
import timm

from models.MAN.MANUtils import print_num_params
from compression.skt_utils import cal_para

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

BACKBONE_MODEL = "vgg"

__all__ = ['vgg19_trans']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

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

def init_MAN():
    """ Initializes the MAN model with the appropriate backbone network
    
        :returns: Initialized model

        :rtype: Object
    """
    # For EfficientNet backbones, download the corresponding model from the Hugging Face repository
    if (BACKBONE_MODEL == "efficientnet-b0"):
        model = MAN(timm.create_model('tf_efficientnet_b0', pretrained=True))
    elif (BACKBONE_MODEL == "efficientnet-b3"):
        model = MAN(timm.create_model('tf_efficientnet_b3', pretrained=True))
    elif (BACKBONE_MODEL == "efficientnet-b5"):
        model = MAN(timm.create_model('tf_efficientnet_b5', pretrained=True))
    # For the VGG-19 backbone, use the VGG 19-layer model (configuration "E") pre-trained on ImageNet
    else:
        model = MAN(make_layers(cfg['E']))
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    return model
    
class MANSolver(Trainer):
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

        self.datasets = {x: Crowd(os.path.join(args.data_dir, x),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.dataset_name, args.cc_50_val, args.cc_50_test, args.is_gray, args.augment_contrast, args.augment_contrast_factor, args.augment_save_location, args.augment_save, x) for x in ['train', 'test']}
            
        self.dataloaders = {x: DataLoader(self.datasets[x],
                                          collate_fn=(train_collate
                                                      if x == 'train' else default_collate),
                                          batch_size=(args.batch_size
                                          if x == 'train' else 1),
                                          shuffle=(True if x == 'train' else False),
                                          num_workers=args.num_workers*self.device_count,
                                          pin_memory=(True if x == 'train' else False))
                            for x in ['train', 'test']}
        self.model = init_MAN()
        self.model.to(self.device)
        
        print_num_params(self.model, self.model)
        cal_para(self.model)

        if ('efficientnet' in BACKBONE_MODEL):
            self.optimizer = optim.RMSprop(self.model.parameters(), lr = args.lr, weight_decay = args.weight_decay, momentum = 0.9)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.start_epoch = 0
        if args.resume:
            suf = args.resume.rsplit('.', 1)[-1]
            if suf == 'tar':
                checkpoint = torch.load(args.resume, self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.start_epoch = checkpoint['epoch']
            elif suf == 'pth':
                self.model.load_state_dict(torch.load(args.resume, self.device))

        self.post_prob = Post_Prob(args.sigma,
                                   args.crop_size,
                                   args.downsample_ratio,
                                   args.background_ratio,
                                   args.use_background,
                                   self.device)
        self.criterion = Bay_Loss(args.use_background, self.device)
        #self.criterion = torch.nn.MSELoss(reduction='sum').cuda()
        self.save_list = Save_Handle(max_num=args.max_model_num)
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.save_all = args.save_all
        self.best_count = 0

    def train(self):
        """ Performs model training
        """
        args = self.args
        
        #ADDED VARIABLE INITIALIZATION FOR WARMUP LEARNING
        self.sched = 0
        
        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-'*5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-'*5)
            self.epoch = epoch
            
            self.train_epoch()
            if epoch % args.val_epoch == 0 and epoch >= args.val_start:
                self.val_epoch()
      
    def train_epoch(self):
        """ Performs a single epoch of model training
        """
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        epoch_start = time.time()
        self.model.train()  # Set model to training mode
        
        #ADDED CODE FOR WARMUP LEARNING   
                
        num_sched = len(self.learning_sched)
        if num_sched != 0 and self.sched < num_sched:
            if (self.epoch + 1) in self.learning_sched:
                self.lr *= 10
                print("Learning rate increased to ", self.lr)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
                self.sched += 1
                        
        # Iterate over data.
        for step, (inputs, points, targets, st_sizes) in enumerate(self.dataloaders['train']):
            inputs = inputs.to(self.device)
            st_sizes = st_sizes.to(self.device)
            gd_count = np.array([len(p) for p in points], dtype=np.float32)
            points = [p.to(self.device) for p in points]
            targets = [t.to(self.device) for t in targets]

            with torch.set_grad_enabled(True):
                outputs, features = self.model(inputs)
                prob_list = self.post_prob(points, st_sizes)
                loss = self.criterion(prob_list, targets, outputs)
                loss_c = 0
                for feature in features:
                    mean_feature = torch.mean(feature, dim=0)
                    mean_sum = torch.sum(mean_feature**2)**0.5
                    cosine = 1 - torch.sum(feature*mean_feature, dim=1) / (mean_sum * torch.sum(feature**2, dim=1)**0.5 + 1e-5)
                    loss_c += torch.sum(cosine)
                loss += loss_c

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                N = inputs.size(0)
                pre_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                res = pre_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(res * res), N)
                epoch_mae.update(np.mean(abs(res)), N)

        logging.info('Epoch {} Train, Loss: {:.2f}, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, epoch_loss.get_avg(), np.sqrt(epoch_mse.get_avg()), epoch_mae.get_avg(),
                             time.time()-epoch_start))
        model_state_dic = self.model.state_dict()
        save_path = os.path.join(self.save_dir, '{}_ckpt.tar'.format(self.epoch))
        torch.save({
            'epoch': self.epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_state_dict': model_state_dic
        }, save_path)
        self.save_list.append(save_path)  # control the number of saved models
        
        #####START HERE####
        
        val_epoch_mse = np.sqrt(epoch_mse.get_avg())
        val_epoch_mae = epoch_mae.get_avg()
        
        if (2.0 * val_epoch_mse + val_epoch_mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = val_epoch_mse
            self.best_mae = val_epoch_mae
            logging.info("save best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                 self.best_mae,
                                                                                 self.epoch))
            if self.save_all:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model_{}.pth'.format(self.epoch)))
                self.best_count += 1
            else:
                torch.save(model_state_dic, os.path.join(self.save_dir, 'best_model.pth'))
        
    def val_epoch(self):
        """ Performs model validation
        """
        epoch_start = time.time()
        self.model.eval()  # Set model to evaluate mode
        epoch_res = []
        # Iterate over data.
        for inputs, count, name in self.dataloaders['test']:
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
                with torch.set_grad_enabled(False):
                    pre_count = 0.0
                    for idx, input in enumerate(input_list):
                        output = self.model(input)[0]
                        pre_count += torch.sum(output)
                res = count[0].item() - pre_count.item()
                epoch_res.append(res)
            else:
                with torch.set_grad_enabled(False):
                    outputs = self.model(inputs)[0]
                    # save_results(inputs, outputs, self.vis_dir, '{}.jpg'.format(name[0]))
                    res = count[0].item() - torch.sum(outputs).item()
                    epoch_res.append(res)

        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
        logging.info('Epoch {} Val, MSE: {:.2f} MAE: {:.2f}, Cost {:.1f} sec'
                     .format(self.epoch, mse, mae, time.time()-epoch_start))

        model_state_dic = self.model.state_dict()
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
        epoch_start = time.time()
        args = args

        datasets = Crowd(os.path.join(args.data_dir, 'test'),
                                  args.crop_size,
                                  args.downsample_ratio,
                                  args.dataset_name, args.cc_50_val, args.cc_50_test, args.is_gray, args.augment_contrast, args.augment_contrast_factor, args.augment_save_location, args.augment_save, method='test')

        dataloader = torch.utils.data.DataLoader(datasets, 1, shuffle=False,
                                                 num_workers=8, pin_memory=False)

        device = torch.device('cuda')
        model = init_MAN()
        model.to(device)
        model.eval()

        model.load_state_dict(torch.load(args.best_model_path, device))
        print(time.time())
        epoch_minus = []
        for inputs, count, name in dataloader:
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
                with torch.set_grad_enabled(False):
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
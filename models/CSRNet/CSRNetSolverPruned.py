import torch
import time
import torch.nn as nn
import torch.optim as optim
import math
import os
from datetime import date

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import time

from tqdm import tqdm
from models.CSRNet.CSRNet import CSRNet
from models.CSRNet.CSRNetDataset import CSRNetDataset
from models.CSRNet.CSRNetPruned import CSRNetPruned

from utilities.pruner import VanillaPruner

from models.CSRNet.CSRNetUtils import save_checkpoint

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

class CSRNetSolverPruned(object):

    def __init__(self, config, paths):
        """ Initializes a CSRNet Solver object

        Arguments:
            config {Object} -- configurations of the model
            paths {Object} -- paths to the resources used by the model  
        """
        self.config = config
        self.paths = paths
        self.original_lr = self.config.lr
        self.scales = [1,1,1,1]
        self.steps = [-1,1,100,150]
        self.workers = 4
        self.start_epoch = 0
        self.print_freq = 30
        self.data = ''

        self.build_model()

    def print_num_params(self, model, name):
        """ Prints the structure of the network and the total number of parameters

            :param Object model: the model to be used
            :param string name: name of the model
        """
        num_params = 0
        for name, param in model.named_parameters():
            if 'transform' in name:
                continue
            num_params += param.data.numel()
        
        print('The number of parameters: ', num_params)

    def build_model(self):
        """ Instantiates the model, loss criterion, and optimizer
        """
        self.model = CSRNetPruned().cuda()
        self.criterion = nn.L1Loss(size_average=False).cuda()
        self.optimizer = optim.SGD(params=self.model.parameters(), lr=self.config.lr, momentum=self.config.momentum, weight_decay=self.config.weight_decay)

        # enable GPU use if available and set to True by user
        if torch.cuda.is_available() and self.config.use_gpu:
            self.model.cuda()
            self.criterion.cuda()

        # print network
        print(self.model)
        print(self.optimizer)
        self.print_num_params(self.model, self.config.model) 

    def start(self, config):
        """ Starts model training
        
            :param Object config: configurations of the model
        """
        global best_prec1
    
        best_prec1 = 1e6
        best_rmse = 1e6
        start_time = time.time()

        folder_name = str(config.model) + ' ' + str(date.today().strftime("%d-%m-%Y") + ' ' + str(time.strftime("%H_%M_%S", time.localtime())))
        self.save_path = os.path.join('./weights', folder_name)

        try:
            os.makedirs(self.save_path)
            print("Directory successfully created")
        except:
            print("Directory already exists")

        self.log_path = os.path.join(self.save_path, 'train_log.txt')
        f = open(self.log_path, "w")
        f.close()

        rule = [
            ['frontend.0.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.2.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.5.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.7.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.10.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.12.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.14.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.17.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.19.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['frontend.21.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['backend.0.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['backend.2.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['backend.4.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['backend.6.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['backend.8.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['backend.10.weight', 'filter', [0.5, 0.7], 'l1norm'],
            ['output_layer.weight', 'filter', [0.5, 0.7], 'l1norm']
            # ['0.weight', 'element', [0.3, 0.5], 'abs']
            # ['1.weight', 'kernel', [0.4, 0.6], 'default'],
            # ['2.weight', 'filter', [0.5, 0.7], 'l2norm']
        ]   


        pruner = VanillaPruner(rule=rule)

        for epoch in range(self.start_epoch, config.num_epochs):
            self.adjust_learning_rate(self.optimizer, epoch)
           
            self.train(self.model, self.criterion, self.optimizer, epoch, self.config, pruner)
            prec1, rmse = self.validate(self.model, self.criterion, self.config)
            
            is_best = prec1 < best_prec1
            
            best_prec1 = min(prec1, best_prec1)
            best_rmse = min(rmse, best_rmse)

            print(' * best MAE {mae:.3f} '
                .format(mae=best_prec1))
            
            print(' * best RMSE {rmse:.3f} '
                .format(rmse=best_rmse))

            save_checkpoint({
                'epoch': epoch + 1,
                'arch': self.paths.pretrained_model,
                'state_dict': self.model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : self.optimizer.state_dict(),
            }, is_best, self.config, self.save_path)

            print('current time: %s' % (time.time() - start_time))

            f = open(self.log_path, "a")
            f.write('current MAE: {mae:.9f}\n'.format(mae=prec1))
            f.write('current RMSE: {rmse:.3f}\n'.format(rmse=rmse))
            f.write(' * best MAE: {mae:.9f}\n'.format(mae=best_prec1))
            f.write(' * best RMSE: {rmse:.3f}\n'.format(rmse=best_rmse))
            f.write('current time: %s\n' % (time.time() - start_time))
            f.close()

    def train(self, model, criterion, optimizer, epoch, config, pruner):
        """ Performs model training
        
            :param Object model: model to be used
            :param Object criterion: criterion to be used
            :param Object optimizer: optimizer to be used
            :param int epoch: starting epoch
            :param Object config: configurations used for training
            :param Object pruner: tool for performing pruning
        """
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        if self.config.dataset == 'Shanghaitech-A':
            self.data = self.paths.shanghaitech_a
        elif self.config.dataset == 'Shanghaitech-B':
            self.data = self.paths.shanghaitech_b
        elif self.config.dataset == 'UCFCC50':
            self.data = self.paths.ucf_cc_50
        elif self.config.dataset == 'UCFQNRF':
            self.data = self.paths.ucf_qnrf
            
        
        print("begin train")
        train_loader = torch.utils.data.DataLoader(
            CSRNetDataset(config, self.data,
                        shuffle=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ]), 
                        train=True, 
                        seen=self.model.seen,
                        batch_size=self.config.batch_size,
                        num_workers=self.workers),
            batch_size=self.config.batch_size)
        print(train_loader)
        print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), self.config.lr))
        
        model.train()
        
        end = time.time()

        if epoch == 0:
            pruner.prune(model=model, stage=0, update_masks=True)
        
        for i,(img, target) in enumerate(tqdm(train_loader)):
            data_time.update(time.time() - end)
            
            img = img.cuda()
            img = Variable(img)
            output = model(img)
            
            target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
            target = Variable(target)
            
            loss = criterion(output, target)
            
            losses.update(loss.item(), img.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  

            pruner.prune(model=model, stage=0, update_masks=False)
            
            batch_time.update(time.time() - end)
            end = time.time()
        
        self.print_num_params(model, self.config.model)   
        
        print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            .format(
            epoch, i, len(train_loader), batch_time=batch_time,
            data_time=data_time, loss=losses))

        f = open(self.log_path, "a")
        f.write('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.9f} ({loss.avg:.9f})\t\n'
            .format(
            epoch, i, len(train_loader), batch_time=batch_time,
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
        print ('begin test')
        test_loader = torch.utils.data.DataLoader(
        CSRNetDataset(config, self.data,
                    shuffle=False,
                    transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ]),  train=False),
        batch_size=self.config.batch_size)    
        
        model.eval()
        
        mae = 0
        rmse = 0
        
        for i,(img, target) in enumerate(test_loader):
            img = img.cuda()
            img = Variable(img)
            output = model(img)
            
            mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
            rmse += (output.data.sum()-target.sum().type(torch.FloatTensor).cuda())**2
            
        mae = mae/len(test_loader)  
        rmse = rmse/len(test_loader)  
        rmse = math.sqrt(rmse)
        print(' * MAE {mae:.3f} '
                .format(mae=mae))

        print(' * RMSE {rmse:.3f} '
                .format(rmse=rmse))

        return mae, rmse    

    def adjust_learning_rate(self, optimizer, epoch):
        """ Sets the learning rate of the model 
        
            :param Object optimizer: optimizer used by the model
            :param int epoch: current epoch number of the model
        """
        self.config.lr = self.original_lr
        
        for i in range(len(self.steps)):
            
            scale = self.scales[i] if i < len(self.scales) else 1
            
            
            if epoch >= self.steps[i]:
                self.config.lr = self.config.lr * scale
                if epoch == self.steps[i]:
                    break
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.config.lr

      
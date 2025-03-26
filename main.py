import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1" # make into list if more than one GPU

import numpy as np
from torch.backends import cudnn
import torch

from models.CSRNet.CSRNetSolver import CSRNetSolver
from models.CSRNet.CSRNetDataset import CSRNetDataset

from models.CAN.CANSolver import CANSolver
from models.CAN.CANSolverSKT import CANSolverSKT
from models.CAN.CANSolverPruned import CANSolverPruned

from models.MAN.MANSolver import MANSolver
from models.MAN.MANSolverSKT import MANSolverSKT
from models.MAN.MANSolverPruned import MANSolverPruned
import argparse

from visualizations.cnn_layer_visualization import CNNLayerVisualization
from visualizations.layer_activation_with_guided_backprop import run_with_model


# TODO Add parameters to the constructor during THS-ST3 so that users can utilize the pipeline without changing the code
class Paths(object):
    def __init__(self):
        """ Initializes a Paths object

        Attributes:
            pretrained_model {string} -- path to the pretrained model for validation, test, or prediction; set to None during training
            weights {string} -- path to the folder containing saved models
            test_results {string} -- path to the folder containing the results of testing
            shanghaitech_a {string} -- path to the folder containing the shanghaitech_a dataset
            shanghaitech_b {string} -- path to the folder containing the shanghaitech_b dataset
            ucf_cc_50 {string} -- path to the folder containing the ucf_cc_50 dataset
            man_shanghaitech_a {string} -- path to the folder containing the preprocessed shanghaitech_a dataset used by MAN
            man_shanghaitech_b {string} -- path to the folder containing the preprocessed shanghaitech_b dataset used by MAN
            man_ucf_cc_50 {string} -- path to the folder containing the preprocessed ucf_cc_50 dataset used by MAN
        """
        self.pretrained_model = None 
        self.weights = './weights'
        self.test_results = './tests'
        self.shanghaitech_a = '../Datasets/ShanghaiTechA/'
        self.shanghaitech_b = '../Datasets/ShanghaiTechB/'
        self.ucf_cc_50 = '../Datasets/UCF-CC-50/folds/'
        self.man_shanghaitech_a = '../Datasets/ShanghaiTechAPreprocessed/'
        self.man_shanghaitech_b = '../Datasets/ShanghaiTechBPreprocessed/'
        self.man_ucf_cc_50 = '../Datasets/UCF-CC-50Preprocessed/folds/'

class Config(object):
    def __init__(self):
        """ Initializes a Config object

        Attributes:
            mode {string} -- determines how the model will be used within the pipeline
            model {string} -- determines what crowd counting model to be used for the session
            dataset {string} -- determines what dataset the crowd counting model will be used on
            learning_rate {double} -- learning rate of the model
            learning_sched {list} -- epoch numbers for implementing warmup learning
            momentum {double} -- momentum of the model
            weight_decay {double} -- weight decay of the model
            num_epochs {int} -- number of epochs for training
            batch_size {int} -- batch size of the model
            use_gpu {bool} -- True if the session will use the GPU; False if the session will use the CPU
            weights {string} -- path to the saved weights of the model
            compression {bool} -- True if model compression is to be implemented; False otherwise
            compression_technique {string} -- model compression technique to be used
            lamb_fsp {double} -- value of the flow of solution procedure for SKT
            lamb_cos {double} -- value of the cosine similarity for SKT
            SKT_teacher_ckpt {string} -- path to the partially trained teacher model for SKT
            SKT_student_ckpt {string} -- path to the partially trained student model for SKT
        """
        self.mode = "Test"                 
        self.model = "MAN"              
        self.dataset = "Shanghaitech-A" # [Shanghaitech-A, Shanghaitech-B, UCFCC50, UCFQNRF] 
        self.cc50_val = 3 # [1, 2, 3, 4, 5]
        self.cc50_test = 5 # [1, 2, 3, 4, 5]

        self.lr = 5e-6
        self.learning_sched = []
        
        self.momentum = 0.95             
        self.weight_decay = 1e-5  
        self.num_epochs = 1200            
        self.batch_size = 1    
        self.use_gpu = True
        self.weights = ""

        self.compression = True
        self.compression_technique = "Pruning" # [Pruning, SKT]
        self.lamb_fsp = None
        self.lamb_cos = None
        self.SKT_teacher_ckpt = None
        self.SKT_student_ckpt = None
        
        print('GPU:', torch.cuda.current_device())
        print('GPU Name:', torch.cuda.get_device_name(torch.cuda.current_device()))
        

def main():
    """ Runs the specified model and model compression technique
    """
    # for faster training
    cudnn.benchmark = True
    config = Config()
    paths = Paths()

    # Prepare uncompressed model
    if (config.compression == False):
        if config.model == "CSRNet":
            solver = CSRNetSolver(config, paths)
            solver.start(config)

        elif config.model == "CAN":
            solver = CANSolver(config, paths)
            solver.start(config)

        elif config.model == "MAN":
            args = parse_args(config, paths)
            solver = MANSolver(args)
            
            if config.mode == "Train":
                solver.setup()
                solver.train()
            elif config.mode == "Test":
                solver.test(args)
    
    else:
        # Prepare pruned model
        if config.compression_technique == "Pruning":
            if config.model == "CAN":
                solver = CANSolverPruned(config, paths)
                solver.start(config)
            
            elif config.model == "MAN":
                args = parse_args(config, paths)
                solver = MANSolverPruned(args)

                if config.mode == "Train":
                    solver.setup()
                    
                    # layer visualization
                    print("Visualizing Layers")
                    Conv2D = [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
                    MaxPool2D = [4,9,18,27,36]
                    ReLU = [1,3,6,8,11,13,15,17,20,22,24,26,29,31,33,35]
                    for i in range(0,37):
                        message = "Visualizing layer " + str(i)
                        layerType = ""
                        print(message)
                        if i in Conv2D:
                            print("Layer: Conv2D")
                            layerType = "Conv2D"
                        elif i in MaxPool2D:
                            print("Layer: MaxPool2D")
                            layerType = "MaxPool2D"
                        else:
                            print("Layer: ReLU")
                            layerType = "ReLU"
                        layer_vis = CNNLayerVisualization(solver.model.features, i, 5)
                        layer_vis.visualise_layer_with_hooks(layerType)
                        print()
                        
                    solver.train()
                elif config.mode == "Test":
                    solver.test(args)
        
        else:
            # Prepare model compressed using SKT
            if config.model == "CAN":
                solver = CANSolverSKT(config, paths)
                solver.start(config)
            
            elif config.model == "MAN":
                args = parse_args(config, paths)
                solver = MANSolverSKT(args)

                if config.mode == "Train":
                    solver.setup()
                    solver.train()
                elif config.mode == "Test":
                    solver.test(args)

def parse_args(config, paths):
    """ Parses additional arguments used by MAN
    
        :param Object config: Config object containing the specified configurations
        :param Object paths: Paths object containing the specified paths
    """
    config = config
    paths = paths
    
    # Dataset paths for MAN
    if (config.dataset == "Shanghaitech-A"):
        dataset_path = paths.man_shanghaitech_a
    elif (config.dataset == "Shanghaitech-B"):
        dataset_path = paths.man_shanghaitech_b
    elif (config.dataset == "UCFCC50"):
            dataset_path = paths.man_ucf_cc_50
    else:
            dataset_path = paths.man_ucf_qnrf
    
    parser = argparse.ArgumentParser(description=config.mode)
    
    # Training details
    parser.add_argument('--model-name', default=config.model, help='the name of the model')
    parser.add_argument('--dataset-name', default=config.dataset, help='the name of the dataset')
    parser.add_argument('--data-dir', default=dataset_path,
                        help='training data directory')
    parser.add_argument('--cc-50-val', default=config.cc50_val, help='fold number to use as validation set for cc50')
    parser.add_argument('--cc-50-test', default=config.cc50_test, help='fold number to use as test set for cc50')
    parser.add_argument('--save-dir', default=paths.weights,
                        help='directory to save models.')
    parser.add_argument('--save-all', type=bool, default=True,
                        help='whether to save all best model')
    parser.add_argument('--lr', type=float, default=config.lr,
                        help='the initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=config.weight_decay,
                        help='the weight decay')
    parser.add_argument('--resume', default=None,
                        help='the path of resume training model')
    parser.add_argument('--max-model-num', type=int, default=2,
                        help='max models num to save ')
    parser.add_argument('--max-epoch', type=int, default=config.num_epochs,
                        help='max training epoch')
    parser.add_argument('--val-epoch', type=int, default=1,
                        help='the num of steps to log training information')
    parser.add_argument('--val-start', type=int, default=600,
                        help='the epoch start to val')
    parser.add_argument('--batch-size', type=int, default=config.batch_size,
                        help='train batch size')
    parser.add_argument('--device', default='1', help='assign device')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='the num of training process')
    parser.add_argument('--is-gray', type=bool, default=False,
                        help='whether the input image is gray')
    parser.add_argument('--crop-size', type=int, default=256,
                        help='the crop size of the train image')
    parser.add_argument('--downsample-ratio', type=int, default=16,
                        help='downsample ratio')

    parser.add_argument('--use-background', type=bool, default=True,
                        help='whether to use background modelling')
    parser.add_argument('--sigma', type=float, default=8.0,
                        help='sigma for likelihood')
    parser.add_argument('--background-ratio', type=float, default=0.15,
                        help='background ratio')
    parser.add_argument('--best-model-path', default=config.weights,
                        help='best model path')
    parser.add_argument('--learning-sched', default = config.learning_sched,
                        help='number of epochs for warmup learning')
    
    # Contrast enhancement
    parser.add_argument('--augment-contrast', type=bool, default=False,
                        help='whether to apply contrast enhancement on images')
    parser.add_argument('--augment-contrast-factor', type=float, default=0.5,
                        help='Contrast enhancement factor')
    parser.add_argument('--augment-save', type=bool, default=True, help='whether to save augmented images')
    parser.add_argument('--augment-save-location', default="", help='save folder of augmented images')
    
    # Model compression
    parser.add_argument('--compression', default = config.compression,
                        help='whether compression is to be implemented')
    parser.add_argument('--compression-technique', default = config.compression_technique,
                        help='compression technique to be used')
    parser.add_argument('--lamb-fsp', default = config.lamb_fsp,
                        help='weight of dense fsp loss')
    parser.add_argument('--lamb-cos', default = config.lamb_fsp,
                        help='weight of cos loss')
    parser.add_argument('--teacher_ckpt', default = config.SKT_teacher_ckpt,
                        help='SKT teacher checkpoint')
    parser.add_argument('--student_ckpt', default = config.SKT_student_ckpt,
                        help='SKT student checkpoint')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
    
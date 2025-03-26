import os
import logging
from datetime import datetime

def setlogger(path):
    """ Sets the logger for the model
    
        :param string path: path to the logger
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(asctime)s %(message)s",
                                     "%m-%d %H:%M:%S")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
    
def print_num_params(model, name):
    """ Prints the number of parameters of the model
    
        :param Object model: model to be used
        :param string name: name of the model
    """
    num_params = 0
    for name, param in model.named_parameters():
        if 'transform' in name:
            continue
        num_params += param.data.count_nonzero()
    
    print("The number of parameters: ", num_params)

class Trainer(object):
    def __init__(self, args):
        """ Initializes a Trainer object
        """
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        setlogger(os.path.join(self.save_dir, 'train.log'))  # set logger
        for k, v in args.__dict__.items():  # save args
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        """ Initializes the datasets, model, loss and optimizer
        """
        pass

    def train(self):
        """ Performs training for one epoch
        """
        pass

class Save_Handle(object):
    def __init__(self, max_num):
        """ Initializes a Save_Handle object
        """
        self.save_list = []
        self.max_num = max_num

    def append(self, save_path):
        """ Adds a new model training checkpoint to the list of checkpoints to be saved 
        
            :param string save_path: path of the checkpoint to be saved
        """
        
        # If the total number of saved checkpoints does not exceed the specified maximum number, add
        # the current checkpoint to the list 
        if len(self.save_list) < self.max_num:
            self.save_list.append(save_path)
        # Otherwise, remove the oldest saved checkpoint before saving the new checkpoint
        else:
            remove_path = self.save_list[0]
            del self.save_list[0]
            self.save_list.append(save_path)
            if os.path.exists(remove_path):
                os.remove(remove_path)

class AverageMeter(object):
    def __init__(self):
        """ Initializes an AverageMeter object
        """
        self.reset()

    def reset(self):
        """ Resets the attributes of the AverageMeter object
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Updates the values of the AverageMeter object
        
            :param int val: new value of the object
            :param int n: amount to be added to the count of the object
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = 1.0 * self.sum / self.count

    def get_avg(self):
        """ Gets the average of the AverageMeter object
        
            :returns: Average as computed by the object

            :rtype: double
        """
        return self.avg

    def get_count(self):
        """ Gets the total count of the AverageMeter object
        
            :returns: number of data instances stored in the AverageMeter object

            :rtype: int
        """
        return self.count
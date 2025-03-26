import torch.nn as nn
import torch
from torchvision import models

class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        """ Initializes a CSRNet object

        Keyword Arguments:
            load_weights {boolean} -- whether pretrained weights are to be loaded {default: False}
        """
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16()
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def forward(self,x):
        """ Implements a forward pass
        
            :param list x: input features of the model

            :returns: Updated model features

            :rtype: list
        """
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        """ Initializes the weights of the model
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def regist_hook(self):
        """ Adds hooks between the teacher and student models
        """
        self.features = []

        def get(model, input, output):
            """ Appends the hooks to the model features
            
                :param Object model: model where the hooks are appended
                :param list input: input features of the model
                :param list output: output features of the model
            """
            # function will be automatically called each time, since the hook is injected
            self.features.append(output.detach())

        for name, module in self._modules['frontend']._modules.items():
            if name in ['1', '4', '9', '16']:
                self._modules['frontend']._modules[name].register_forward_hook(get)
        for name, module in self._modules['backend']._modules.items():
            if name in ['1', '7']:
                self._modules['backend']._modules[name].register_forward_hook(get)

def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    """ Creates the layers of the model
    
        :param list cfg: number of channels per layer of the model
        :param int in_channels: number of input channels
        :param boolean batch_norm: whether batch normalization is to be implemented
        :param boolean dilation: whether dilation is to be implemented {default: False}

        :returns: Sequential container storing the layers of the model

        :rtype: nn.Sequential
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)                
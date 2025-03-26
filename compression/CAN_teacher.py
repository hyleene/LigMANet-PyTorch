import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models

class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        """ Initializes a ContextualModule object
        
            :param list features: feature values of the module
            :param int out_features: final number of channels the features after passing through the bottleneck layer
            :param list sizes: sizes used to scale the modules
        """
        super(ContextualModule, self).__init__()
        self.scales = []
        self.scales = nn.ModuleList([self._make_scale(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * 2, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.weight_net = nn.Conv2d(features,features,kernel_size=1)

    def __make_weight(self,feature,scale_feature):
        """ Creates the model weights
        
            :param list feature: original feature values of the model
            :param list scale_feature: scaled feature values of the model

            :returns: result of the sigmoid function on the features
            :rtype: double
        """
        weight_feature = feature - scale_feature
        return F.sigmoid(self.weight_net(weight_feature))

    def _make_scale(self, features, size):
        """ Scales the pooling and convolutional layers of the model
        
            :param list features: feature values of the model
            :param int size: target output size of the layer

            :returns: Sequential container storing the prior and convolutional layers

            :rtype: nn.Sequential
        """
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        """ Implements the forward pass of the model features
        
            :param list feats: features of the model

            :returns: Result of the ReLU function on the bottleneck layer

            :rtype: double
        """
        h, w = feats.size(2), feats.size(3)
        multi_scales = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.scales]
        weights = [self.__make_weight(feats,scale_feature) for scale_feature in multi_scales]
        overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])]+ [feats]
        bottle = self.bottleneck(torch.cat(overall_features, 1))
        return self.relu(bottle)

class CANNet(nn.Module):
    def __init__(self, load_weights=False):
        """ Initializes a CANNet object
        
            :param boolean load_weights: whether pretrained weights are to be loaded
        """
        super(CANNet, self).__init__()
        self.seen = 0
        self.context = ContextualModule(512, 512)
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,batch_norm=True, dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                # self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]
        self.features = []

    def forward(self,x):
        """ Implements the forward pass of the entire model
        
            :param list x: input features of the model

            :returns: Updated features of the model after one forward pass

            :rtype: list
        """
        self.features = []
        x = self.frontend(x)
        x = self.context(x)
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
        for name, module in self._modules['context']._modules.items():
            if name in ['relu']:
                self._modules['context']._modules[name].register_forward_hook(get)
        for name, module in self._modules['backend']._modules.items():
            if name in ['2', '11']:
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
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

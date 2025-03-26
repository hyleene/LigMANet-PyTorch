import torch.nn as nn
import torch
from torch.nn import functional as F
from torchvision import models

class ContextualModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        """ Initializes a ContextualModule object
        
            :param list features: feature values of the module
            :param int outfeatures: final number of channels the features after passing through the bottleneck layer
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
            :returns: Result of the sigmoid function on the features

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

# Channel numbers for different channel preservation rates
channel_nums = [[52, 103, 205, 410],  # 20% removed (4/5 preservation)
                [39, 77, 154, 308],  # 40% removed (3/5 preservation)
                [26, 52, 103, 205],  # 60% removed (2/5 preservation)
                [13, 26, 52, 103]  # 80% removed (1/5 preservation)
                ]

class CANNet(nn.Module):
    def __init__(self, ratio=4, transform=True):
        """ Initializes a CANNet object
        
            :param int ratio: offset value for choosing the channel preservation rate
            :param boolean transform: whether transform layers are to be added to the model
        """
        super(CANNet, self).__init__()
        self.seen = 0
        self.transform = transform
        channel = channel_nums[ratio-2]

        # front-end
        self.conv0_0 = conv_layers(3, channel[0])
        if self.transform:
            self.transform0_0 = feature_transform(channel[0], 64)
        self.conv0_1 = conv_layers(channel[0], channel[0])

        self.pool0 = pool_layers()
        if transform:
            self.transform1_0 = feature_transform(channel[0], 64)
        self.conv1_0 = conv_layers(channel[0], channel[1])
        self.conv1_1 = conv_layers(channel[1], channel[1])

        self.pool1 = pool_layers()
        if transform:
            self.transform2_0 = feature_transform(channel[1], 128)
        self.conv2_0 = conv_layers(channel[1], channel[2])
        self.conv2_1 = conv_layers(channel[2], channel[2])
        self.conv2_2 = conv_layers(channel[2], channel[2])

        self.pool2 = pool_layers()
        if transform:
            self.transform3_0 = feature_transform(channel[2], 256)
        self.conv3_0 = conv_layers(channel[2], channel[3])
        self.conv3_1 = conv_layers(channel[3], channel[3])
        self.conv3_2 = conv_layers(channel[3], channel[3])

        # context
        self.context = ContextualModule(channel[3], channel[3])
        if transform:
            self.transform_context = ContextualModule(channel[3], 512)

        # scales_0 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Conv2d(channel[3], channel[3], kernel_size=1, bias=False))
        # scales_1 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Conv2d(channel[3], channel[3], kernel_size=1, bias=False))
        # scales_2 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(3, 3)), nn.Conv2d(channel[3], channel[3], kernel_size=1, bias=False))
        # scales_3 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(channel[3], channel[3], kernel_size=1, bias=False))
        # self.scales = []
        # self.scales = nn.ModuleList([scales_0, scales_1, scales_2, scales_3])
        # self.bottleneck = nn.Conv2d(channel[3] * 2, channel[3], kernel_size=1)
        # self.relu = nn.ReLU()
        # self.weight_net = nn.Conv2d(channel[3],channel[3],kernel_size=1)
        # if transform:
        #     self.transform_context = context_transform(channel[3], 512)

        # back-end
        self.conv4_0 = conv_layers(channel[3], channel[3], batch_norm=True, dilation=2)
        if transform:
            self.transform4_0 = feature_transform(channel[3], 512, batch_norm=True)
        self.conv4_1 = conv_layers(channel[3], channel[3], batch_norm=True, dilation=2)
        self.conv4_2 = conv_layers(channel[3], channel[3], batch_norm=True, dilation=2)
        self.conv4_3 = conv_layers(channel[3], channel[2], batch_norm=True, dilation=2)
        if transform:
            self.transform4_3 = feature_transform(channel[2], 256, batch_norm=True)
        self.conv4_4 = conv_layers(channel[2], channel[1], batch_norm=True, dilation=2)
        self.conv4_5 = conv_layers(channel[1], channel[0], batch_norm=True, dilation=2)

        self.conv5_0 = nn.Conv2d(channel[0], 1, kernel_size=1)

        self._initialize_weights()
        self.features = []

    def forward(self,x):
        """ Implements the forward pass of the entire model
        
            :param list x: input features of the model

            :returns: Updated features of the model after one forward pass

            :rtype: list
        """
        self.features = []

        # front-end
        x = self.conv0_0(x)
        if self.transform:
            self.features.append(self.transform0_0(x))
        x = self.conv0_1(x)

        x = self.pool0(x)
        if self.transform:
            self.features.append(self.transform1_0(x))
        x = self.conv1_0(x)
        x = self.conv1_1(x)

        x = self.pool1(x)
        if self.transform:
            self.features.append(self.transform2_0(x))
        x = self.conv2_0(x)
        x = self.conv2_1(x)
        x = self.conv2_2(x)

        x = self.pool2(x)
        if self.transform:
            self.features.append(self.transform3_0(x))
        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)

        # context
        x = self.context(x)
        if self.transform:
            self.features.append(self.transform_context(x))
        # x = contextual_forward(x, self.scales, self.bottleneck, self.relu, self.weight_net)
        # if self.transform:
        #     self.features.append(self.transform_context(x))

        # back-end
        x = self.conv4_0(x)
        if self.transform:
            self.features.append(self.transform4_0(x))
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        if self.transform:
            self.features.append(self.transform4_3(x))
        x = self.conv4_4(x)
        x = self.conv4_5(x)

        x = self.conv5_0(x)

        self.features.append(x)

        if self.training is True:
            return self.features
        return x

    def _initialize_weights(self):
        """ Initializes the weights of the model
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.normal_(m.weight, std=0.01)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

def context_transform(inp, oup):
    """ Transforms the input features according to the context
    
        :param int inp: number of input channels
        :param int outp: number of output channels

        :returns: Sequential counter containing the list of rescaled features

        :rtype: nn.Sequential
    """
    scales_0 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Conv2d(inp, oup, kernel_size=1, bias=False))
    scales_1 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(2, 2)), nn.Conv2d(inp, oup, kernel_size=1, bias=False))
    scales_2 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(3, 3)), nn.Conv2d(inp, oup, kernel_size=1, bias=False))
    scales_3 = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(6, 6)), nn.Conv2d(inp, oup, kernel_size=1, bias=False))
    bottleneck = nn.Conv2d(inp * 2, oup, kernel_size=1)
    relu = nn.ReLU()
    weight_net = nn.Conv2d(inp,oup,kernel_size=1)
    return nn.Sequential(nn.ModuleList([scales_0, scales_1, scales_2, scales_3]), bottleneck, relu, weight_net)
    
def contextual_forward(feats, scales, bottleneck, relu, weight_net):
    """ Implements the forward pass for the contextual module
    
        :param list feats: features of the model
        :param list scales: scales of the model
        :param Object bottleneck: bottleneck layer of the model
        :param Object relu: ReLU function to be used on the model
        :param Object weight_net: convolutional layer for creating the weights of the model

        :returns: result of the ReLU function on the bottleneck layer

        :rtype: double
    """
    h, w = feats.size(2), feats.size(3)
    multi_scales = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in scales]
    weights = [make_weight(feats,scale_feature,weight_net) for scale_feature in multi_scales]
    overall_features = [(multi_scales[0]*weights[0]+multi_scales[1]*weights[1]+multi_scales[2]*weights[2]+multi_scales[3]*weights[3])/(weights[0]+weights[1]+weights[2]+weights[3])]+ [feats]
    bottle = bottleneck(torch.cat(overall_features, 1))
    return relu(bottle)

def make_weight(feature,scale_feature,weight_net):
    """ Creates weights based on the model features
        
        :param list feature: feature of the model
        :param list scale_feature: scaled feature of the model
        :param Object weight_net: convolutional layer for creating the weights of the model

        :returns: result of the sigmoid function on the weight_net layer

        :rtype: double
    """
    weight_feature = feature - scale_feature
    return F.sigmoid(weight_net(weight_feature))

def conv_layers(inp, oup, batch_norm=False, dilation=False):
    """ Creates the convolutional layers of the model
    
        :param int inp: number of input channels
        :param int outp: number of output channels
        :param boolean batch_norm: whether batch normalization is to be used
        :param boolean dilation: whether dilation is to be used

        :returns: Sequential counter containing the convolutional layers

        :rtype: nn.Sequential
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    if batch_norm == False:
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True)
        )


def feature_transform(inp, oup, batch_norm=False):
    """ Performs feature transformation on the model
    
        :param int inp: number of input channels
        :param int outp: number of output channels
        :param boolean batch_norm: whether batch normalization is to be used

        :returns: Sequential counter containing the convolutional layers

        :rtype: nn.Sequential
    """
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    if batch_norm == False:
        layers += [conv2d, relu]
    else:
        layers += [conv2d, nn.BatchNorm2d(oup), relu]
    return nn.Sequential(*layers)


def pool_layers(ceil_mode=True):
    """ Performs pooling on the layers of the model
        
        :param boolean ceil_mode: whether ceil is used to calculate the output shape

        :returns: Pooling layer

        :rtype: nn.MaxPool2d
    """
    return nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
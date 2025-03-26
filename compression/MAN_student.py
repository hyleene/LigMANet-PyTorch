import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from typing import Optional, List
from torch import Tensor
from torch.nn import Parameter

# Channel numbers for different channel preservation rates
# ORIGINAL: 64, 128, 256, 512
channel_nums = [[57, 115, 230, 461],  # 9/10
                [51, 102, 205, 410],  # 4/5
                [45, 90, 179, 358],  # 7/10
                [38, 77, 154, 307],  # 3/5
                [32, 64, 128, 256],  # 1/2
                [21, 43, 85, 171],  # 1/3
                [16, 32, 64, 128],  # 1/4
                [13, 26, 51, 102],  # 1/5
                [11, 21, 43, 85],   # 1/6
                [9, 18, 37, 73],     # 1/7
                [4, 8, 16, 32]      # 1/8
               ]

class LearnableGlobalLocalMultiheadAttention(nn.Module):
    NUM_WEIGHTS = 9
    def __init__(
            self, embed_dim, num_heads, dropout=0.):
        """ Initializes a LearnableGlobalLocalMultiheadAttention object
        
        Arguments:
            embed_dim {int} -- number of dimensions
            num_heads {int} -- number of heads
        Keyword Arguments:
            dropout {double} -- dropout value
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_k = self.bias_v = None
        self.reset_parameters()

    def reset_parameters(self):
        """ Resets the values of the parameters
        """
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
            
   # global
    def in_proj_global_q(self, query):
        """ Searches the specified query across the entire transformer
        
            :param list query: query value

            :returns: Linear transformation of the specified query

            :rtype: torch.Tensor
        """
        return self._in_proj(query, start=0, end=self.embed_dim)

    def in_proj_global_k(self, key):
        """ Searches the specified key across the entire transformer
        
            :param list key: key value

            :returns: Linear transformation of the specified key

            :rtype: torch.Tensor
        """
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_global_v(self, value):
        """ Searches the specified value across the entire transformer
        
            :param list value: value of the specified value

            :returns: Linear transformation of the specified value

            :rtype: torch.Tensor
        """
        return self._in_proj(value, start=2 * self.embed_dim, end=3 * self.embed_dim)

    # local left
    def in_proj_local_left_q(self, query):
        """ Searches the specified query across the local left of the transformer
        
            :param list query: query value

            :returns: Linear transformation of the specified query

            :rtype: torch.Tensor
        """
        return self._in_proj(query, start=3 * self.embed_dim, end=4 * self.embed_dim)

    def in_proj_local_left_k(self, key):
        """ Searches the specified key across the local left of the transformer
        
            :param list key: key value

            :returns: Linear transformation of the specified key

            :rtype: torch.Tensor
        """
        return self._in_proj(key, start=4 * self.embed_dim, end=5 * self.embed_dim)

    # local right
    def in_proj_local_right_q(self, query):
        """ Searches the specified query across the local right of the transformer
        
            :param list query: query value

            :returns: Linear transformation of the specified query

            :rtype: torch.Tensor
        """
        return self._in_proj(query, start=5 * self.embed_dim, end=6 * self.embed_dim)

    def in_proj_local_right_k(self, key):
        """ Searches the specified key across the local right of the transformer
        
            :param list key: key value

            :returns: Linear transformation of the specified key

            :rtype: torch.Tensor
        """
        return self._in_proj(key, start=6 * self.embed_dim, end=7 * self.embed_dim)

    # local right
    def in_proj_local_q(self, query):
        """ Searches the specified query across the local area of the transformer
        
            :param list query: query value

            :returns: Linear transformation of the specified query

            :rtype: torch.Tensor
        """
        return self._in_proj(query, start=7 * self.embed_dim, end=8 * self.embed_dim)

    def in_proj_local_k(self, key):
        """ Searches the specified key across the local area of the transformer
        
            :param list key: key value

            :returns: Linear transformation of the specified key

            :rtype: torch.Tensor
        """
        return self._in_proj(key, start=8 * self.embed_dim, end=9 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        """ Searches the specified input across a specified range within the transformer
        
        Arguments:
            input {list} -- input value

        Keyword Arguments:
            start {int} -- start of the search space {default: 0}
            end {int} -- end of the search space {default: None}

        Returns:
            torch.Tensor -- linear transformation of the specified input according to the weight and bias
        """
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def prepare_local_masking(self, q_left, k_left, q_right, k_right, shape):
        """ Performs local masking on the specified area of the transformer
        
            :param torch.Tensor q_left: query in the local left of the transformer
            :param torch.Tensor k_left: key in the local left of the transformer
            :param torch.Tensor q_right: query in the local right of the transformer
            :param torch.Tensor k_right: key in the local right of the transformer
            :param list shape: shape of the transformer

            :returns: Local mask for the specified area of the transformer

            :rtype: torch.Tensor
        """
        left_attn_weights = torch.bmm(q_left, k_left.transpose(1, 2))
        right_attn_weights = torch.bmm(q_right, k_right.transpose(1, 2))

        left_size = left_attn_weights.size()
        src_len = left_size[2]

        triu = torch.ones(src_len, src_len, device=q_left.device, dtype=q_left.dtype).triu_()
        mini_triu = torch.ones(shape[1], shape[1], device=q_left.device, dtype=q_left.dtype).triu_()
        mini_triu = mini_triu.repeat(shape[0], shape[0])
        triu = (triu * mini_triu).unsqueeze_(0)

        left_softmax = F.softmax(left_attn_weights, dim=-1)
        right_softmax = F.softmax(right_attn_weights, dim=-1)

        local_mask = self.compute_lrmask2localmask(left_softmax, right_softmax, triu)

        return local_mask

    def compute_lrmask2localmask(self, left_softmax, right_softmax, triu):
        """ Computes for the local mask on the specified area of the transformer
        
            :param torch.Tensor left_softmax: result of softmax function on the local left of the tensor
            :param torch.Tensor right_softmax: result of softmax function on the local right of the tensor
            :param torch.Tensor triu: upper triangular of the local area of the transformer

            :returns: local mask for the specified area of the transformer

            :rtype: torch.Tensor
        """
        triu_t = triu.transpose(1,2)
        left_mask = torch.matmul(left_softmax, triu)
        right_mask = torch.matmul(right_softmax, triu_t)
        bw_left_mask = torch.matmul(left_softmax, triu_t)
        bw_right_mask = torch.matmul(right_softmax, triu)

        fw_mask = left_mask * right_mask
        bw_mask = bw_left_mask * bw_right_mask
        local_mask = fw_mask + bw_mask
        return local_mask

    def forward(self, query, key, shape, value):
        """ Performs the forward pass on the attention module
        
            :param list query: query value
            :param list key: key value
            :param list shape: shape of the attention module
            :param list value: value of the value

            :returns:
                - (:py:class:`torch.Tensor`) - linear transformation of attention values 
                - (:py:class:`torch.Tensor`) - aggregate of all local attention masks
        """
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        q = self.in_proj_global_q(query)
        k = self.in_proj_global_k(key)
        v = self.in_proj_global_v(value)
        q_left = self.in_proj_local_left_q(query)
        k_left = self.in_proj_local_left_k(key)
        q_right = self.in_proj_local_right_q(query)
        k_right = self.in_proj_local_right_k(key)
        q_local = self.in_proj_local_q(query)
        k_local = self.in_proj_local_k(key)

        q = q*self.scaling
        q_local = q_local * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_local = q_local.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_local = k_local.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_left = k_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k_right = k_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_left = q_left.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        q_right = q_right.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        global_attn_weights = torch.bmm(q, k.transpose(1, 2))
        local_attn_weights = torch.bmm(q_local, k_local.transpose(1, 2))

        local_att_mask = self.prepare_local_masking(q_left, k_left, q_right, k_right, shape)
        masked_local_attn_weights = local_attn_weights * local_att_mask

        attn_weights = 0.1 * global_attn_weights + masked_local_attn_weights

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        consistent_mask = torch.sum(local_att_mask, dim=0)

        return attn, consistent_mask
    

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        """ Initializes a TransformerEncoder object
        
        Arguments:
            encoder_layer {Object} -- specified layer
            num_layers {int} -- number of layers

        Keyword Arguments:
            norm {list} -- layer normalization function on the input {default: None}
        """
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, shape,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """ Performs the forward pass for the transformer encoder

            :param list src: input features of the encoder
            :param list shape: shape of the encoder
            :param torch.Tensor mask: mask applied to the input features
            :param torch.Tensor src_key_padding_mask: additional padding to be applied to the mask
            :param torch.Tensor pos: position of the mask

            :returns:
                - (:py:class:`list`) - output of the encoder after the forward pass
                - (:py:class:`list`) - feature values of the encoder
        """
        output = src
        features = []

        for layer in self.layers:
            output, consistent_feature = layer(output, shape, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            features.append(consistent_feature)

        if self.norm is not None:
            output = self.norm(output)

        return output, features
        

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        """ Initializes a TransformerEncoderLayer object
        
        Arguments:
            d_model {int} -- number of dimensions
            nhead {int} -- number of heads

        Keyword Arguments:
            dim_feedforward {int} -- number of dimensions of the feed forward layer {default: 2048}
            dropout {double} -- value of dropout {default: 0.1}
            activation {string} -- activation function to be used {default: "relu"}
            normalize_before {boolean} -- whether normalization is to be performed before the forward pass {default: False}
        """
        super().__init__()
        self.self_attn = LearnableGlobalLocalMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """ Embeds the specified position on the tensor
        
            :param torch.Tensor tensor: specified tensor
            :param torch.Tensor pos: position to be embedded on the tensor

            :returns: Tensor with embedded position

            :rtype: torch.Tensor 
        """
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src, shape,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        """ Performs normalization after the forward pass
        
            :param list src: input features of the encoder layer
            :param list shape: shape of the encoder layer
            :param torch.Tensor src_mask: mask applied to the input features
            :param torch.Tensor src_key_padding_mask: additional padding to be applied to the mask
            :param torch.Tensor pos: position of the mask

            :returns:
                - (:py:class:`list`) - output of the encoder after the forward pass
                - (:py:class:`list`) - feature values of the encoder after applying the mask
        """
        q = k = self.with_pos_embed(src, pos)

        src2, mask = self.self_attn(q, k, shape, src)
        feature = torch.squeeze(src, dim=1)
        consistent_feature = torch.matmul(mask, feature)

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, consistent_feature

    def forward_pre(self, src, shape,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """ Performs normalization before the forward pass
        
            :param list src: input features of the encoder layer
            :param list shape: shape of the encoder layer
            :param torch.Tensor src_mask: mask applied to the input features
            :param torch.Tensor src_key_padding_mask: additional padding to be applied to the mask
            :param torch.Tensor pos: position of the mask

            :returns:
                - (:py:class:`list`) - output of the encoder after the forward pass
                - (:py:class:`list`) - feature values of the encoder after applying the mask
        """
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        ssrc2, mask = self.self_attn(q, k, shape, src)
        feature = torch.squeeze(src, dim=1)
        consistent_feature = torch.matmul(mask, feature)

        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src, consistent_feature

    def forward(self, src, shape,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """ Implements the forward pass of the transformer encoder layer

            :param list src: input features of the encoder layer
            :param list shape: shape of the encoder layer
            :param torch.Tensor src_mask: mask applied to the input features
            :param torch.Tensor src_key_padding_mask: additional padding to be applied to the mask
            :param torch.Tensor pos: position of the mask

            :returns: Encoder output and feature values after the forward pass

            :rtype: Object
        """
        if self.normalize_before:
            return self.forward_pre(src, shape, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, shape, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    """ Creates copies of the specified modules
    
    Arguments:
        module {Object} -- module to be copied
        N {int} -- range of modules to be copied

    Returns:
        nn.ModuleList -- copied modules
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """ Returns an activation function given a string
    
    Arguments:
        activation {string} -- name of specified activation function

    Returns:
        Object -- specified activation function
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MAN(nn.Module):
    def __init__(self, features, ratio, transform=True):
        """ Initializes a MAN object
        
        Arguments:
            features {list} -- input features of the model
            ratio {int} -- offset corresponding to the channel preservation rate of the model

        Keyword Arguments:
            transform {boolean} -- whether transform layers are to be added
        """
        super(MAN, self).__init__()
        self.seen = 0
        self.transform = transform
        channel = channel_nums[ratio]
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
        self.conv2_3 = conv_layers(channel[2], channel[2])

        self.pool2 = pool_layers()
        if transform:
            self.transform3_0 = feature_transform(channel[2], 256)

        self.conv3_0 = conv_layers(channel[2], channel[3])
        self.conv3_1 = conv_layers(channel[3], channel[3])
        self.conv3_2 = conv_layers(channel[3], channel[3])
        self.conv3_3 = conv_layers(channel[3], channel[3])

        self.pool3 = pool_layers()
        if transform:
            self.transform4_0 = feature_transform(channel[3], 512)

        self.conv4_0 = conv_layers(channel[3], channel[3])
        self.conv4_1 = conv_layers(channel[3], channel[3])
        self.conv4_2 = conv_layers(channel[3], channel[3])
        self.conv4_3 = conv_layers(channel[3], channel[3])

        self.pool4 = pool_layers()
        if transform:
            self.transform5_0 = feature_transform(channel[3], 512)

        self._initialize_weights()
        self.features = []

        d_model = channel[3]
            
        nhead = 2
        num_layers = 4
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        if_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder = TransformerEncoder(encoder_layer, num_layers, if_norm)
        self.reg_layer_0 = nn.Sequential(
            nn.Conv2d(d_model, channel[2], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel[2], channel[1], kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel[1], 1, 1)
        )

    def forward(self, x):
        """ Implements the forward pass of the model
            
            :param list x: input to the model

            :returns:
                - (:py:class:`torch.Tensor`) - ReLU transformation of the input
                - (:py:class:`list`) - updated features of the model
        """
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
 
        self.features = []

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
        x = self.conv2_3(x)

        x = self.pool2(x)
        if self.transform:
            self.features.append(self.transform3_0(x))

        x = self.conv3_0(x)
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)

        if self.transform:
            self.features.append(self.transform4_0(x))
        
        x = self.conv4_0(x)
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)

        if self.transform:
            self.features.append(self.transform5_0(x))

        self.features.append(x)
        
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, features = self.encoder(x, (h,w))   # transformer
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        
        x = F.upsample_bilinear(x, size=(rh, rw))
        x = self.reg_layer_0(x)   # regression head

        # if self.training is True:
        #     return self.features, features
        
        return torch.relu(x), features
    
    def _initialize_weights(self):
        """ Initializes the weights of the model
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_normal_(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
def conv_layers(inp, oup, dilation=False):
    """ Creates the convolutional layers of the model
    
        :param int inp: number of input channels
        :param int outp: number of output channels
        :param boolean dilation: whether dilation is to be used

        :returns: Sequential counter containing the convolutional layers

        :rtype: nn.Sequential
    """
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, padding=d_rate, dilation=d_rate),
        nn.ReLU(inplace=True)
    )


def feature_transform(inp, oup):
    """ Performs feature transformation on the model
    
        :param int inp: number of input channels
        :param int outp: number of output channels

        :returns: Sequential storing the updated layers

        :rtype: nn.Sequential
    """
    conv2d = nn.Conv2d(inp, oup, kernel_size=1)  # no padding
    relu = nn.ReLU(inplace=True)
    layers = []
    layers += [conv2d, relu]
    return nn.Sequential(*layers)


def pool_layers(ceil_mode=True):
    """ Performs pooling on the layers of the model
        
        :param boolean ceil_mode: whether ceil is used to calculate the output shape

        :returns: Pooling layer

        :rtype: nn.MaxPool2d
    """
    return nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=ceil_mode)
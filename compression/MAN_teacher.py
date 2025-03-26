import torch.utils.model_zoo as model_zoo
import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
from typing import Optional, List
from torch import Tensor
from torch.nn import Parameter

cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

__all__ = ['vgg19_trans']
model_urls = {'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

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
        Objec -- specified activation function
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class MAN(nn.Module):
    def __init__(self, features):
        """ Initializes a MAN object
        
        Arguments:
            features {list} -- input features of the model
        """
        super(MAN, self).__init__()
        # self.features = features
        self.features = []
        self.frontend = features

        d_model = 512
            
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
            nn.Conv2d(d_model, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )
    
    def forward(self, x):
        """ Implements the forward pass of the model
            
            :param list x: input to the model

            :returns:
                - (:py:class:`torch.Tensor`) - ReLU transformation of the input
                - (:py:class:`list`) - updated features of the model
        """
        self.features = []
        b, c, h, w = x.shape
        rh = int(h) // 16
        rw = int(w) // 16
 
        x = self.frontend(x)   # vgg network
        
        bs, c, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        x, features = self.encoder(x, (h,w))   # transformer
        x = x.permute(1, 2, 0).view(bs, c, h, w)
        
        x = F.upsample_bilinear(x, size=(rh, rw))
        x = self.reg_layer_0(x)   # regression head
        return torch.relu(x), features
        # return x, features
    
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
        for name in self.features:
            if name in ['1', '4', '9', '18', '27', '36']:
                self.features[name].register_forward_hook(get)

    

def init_MAN():
    """Initializes the MAN model
    
    Returns:
        Object -- initialized MAN model
    """
    model = MAN(make_layers(cfg['E']))
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

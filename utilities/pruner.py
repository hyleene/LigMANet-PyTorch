import re
import math
import torch
import tensorflow as tf
from collections.abc import Iterable


def prune_vanilla_elementwise(param, sparsity, fn_importance=lambda x: xs.abs()):
    """ Element-wise vanilla pruning
    
        :param torch.Tensor param: weight of conv/fc layer
        :param float sparsity: pruning sparsity
        :param function fn_importance: inputs 'param' and returns the importance of
                                    each position in 'param'

        :returns: mask for zeros

        :rtype: torch.ByteTensor
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).bool()
    num_el = param.numel()
    importance = fn_importance(param)
    num_pruned = int(math.ceil(num_el * sparsity))
    num_stayed = num_el - num_pruned
    if sparsity <= 0.5:
        _, topk_indices = torch.topk(importance.view(num_el), k=num_pruned,
                                     dim=0, largest=False, sorted=False)
        mask = torch.zeros_like(param).bool()
        param.view(num_el).index_fill_(0, topk_indices, 0)
        mask.view(num_el).index_fill_(0, topk_indices, 1)
    else:
        thr = torch.min(torch.topk(importance.view(num_el), k=num_stayed,
                                   dim=0, largest=True, sorted=False)[0])
        mask = torch.lt(importance, thr)
        param.masked_fill_(mask, 0)
    return mask


def prune_vanilla_kernelwise(param, sparsity, fn_importance=lambda x: x.norm(1, -1)):
    """ Kernel-wise vanilla pruning, the importance determined by L1 norm
    
        :param torch.Tensor param: weight of conv/fc layer
        :param float sparsity: pruning sparsity
        :param function fn_importance: inputs 'param' as size (param.size(0) * param.size(1), -1) and
                                    returns the importance of each kernel in 'param

        :returns: mask for zeros

        :rtype: torch.ByteTensor
    """
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).bool()
    num_kernels = param.size(0) * param.size(1)
    param_k = param.view(num_kernels, -1)
    param_importance = fn_importance(param_k)
    num_pruned = int(math.ceil(num_kernels * sparsity))
    _, topk_indices = torch.topk(param_importance, k=num_pruned,
                                 dim=0, largest=False, sorted=False)
    mask = torch.zeros_like(param).bool()
    mask_k = mask.view(num_kernels, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask


def prune_vanilla_filterwise(sparsity, param, fn_importance=lambda x: x.norm(1, -1)):
    """ Filter-wise vanilla pruning, the importance determined by L1 norm
    
        :param torch.Tensor param: weight of conv/fc layer
        :param float sparsity: pruning sparsity
        :param function fn_importance: inputs 'param' as size (param.size(0), -1) and
                                returns the importance of each filter in 'param'

        :returns: mask for zeros

        :type: torch.ByteTensor
    """
    assert param.dim() >= 3
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        return torch.zeros_like(param).bool()
    num_filters = param.size(0)
    param_k = param.view(num_filters, -1)
    param_importance = fn_importance(param_k)
    num_pruned = int(math.ceil(num_filters * sparsity))
    _, topk_indices = torch.topk(param_importance, k=num_pruned,
                                 dim=0, largest=False, sorted=False)
    mask = torch.zeros_like(param).bool()
    mask_k = mask.view(num_filters, -1)
    param_k.index_fill_(0, topk_indices, 0)
    mask_k.index_fill_(0, topk_indices, 1)
    return mask


class VanillaPruner(object):

    def __init__(self, rule=None):
        """ Initializes a VanillaPruner object
        
        Keyword Arguments: 
            rule {str} -- path to the rule file {default: None}
        """
        if rule:
            if isinstance(rule, str):
                content = map(lambda x: x.split(), open(rule).readlines())
                content = filter(lambda x: len(x) == 3, content)
                rule = list(map(lambda x: (x[0], x[1], list(map(float, x[2].split(',')))), content))
            for r in rule:
                if not isinstance(r[2], Iterable):
                    assert isinstance(r[2], float) or isinstance(r[2], int)
                    r[2] = [float(r[2])]
                if len(r) == 3:
                    r.append('default')
                granularity = r[1]
                if granularity == 'element':
                    r.append(prune_vanilla_elementwise)
                elif granularity == 'kernel':
                    r.append(prune_vanilla_kernelwise)
                elif granularity == 'filter':
                    r.append(prune_vanilla_filterwise)
                else:
                    raise NotImplementedError

        self.rule = rule

        self.masks = dict()

        # print("=" * 89)
        # if self.rule:
        #     print("Initializing Vanilla Pruner with rules:")
        #     for r in self.rule:
        #         print(r[:-1])
        # else:
        #     print("Initializing Vanilla Pruner WITHOUT rules")
        # print("=" * 89)

    def load_state_dict(self, state_dict, replace_rule=True):
        """ Recovers pruner
        
            :param dictionary state_dict: a dictionary containing a whole state of the Pruner
            :param boolean replace_rule: whether to use rule settings in 'state_dict'
        """
        if replace_rule:
            self.rule = state_dict['rule']
            for r in self.rule:
                granularity = r[1]
                if granularity == 'element':
                    r.append(prune_vanilla_elementwise)
                elif granularity == 'kernel':
                    r.append(prune_vanilla_kernelwise)
                elif granularity == 'filter':
                    r.append(prune_vanilla_filterwise)
                else:
                    raise NotImplementedError
        self.masks = state_dict['masks']
        print("=" * 89)
        print("Customizing Vanilla Pruner with rules:")
        for r in self.rule:
            print(r[:-1])
        print("=" * 89)

    def state_dict(self):
        """ Returns a dictionary containing a whole state of the Pruner
        
            :returns: A dictionary containing a whole state of the Pruner

            :rtype: dictionary
        """
        state_dict = dict()
        state_dict['rule'] = [r[:-1] for r in self.rule]
        state_dict['masks'] = self.masks
        return state_dict

    def prune_param(self, param, param_name, stage=0, verbose=False):
        """ Prune parameters
        
            :param torch.Tensor param: parameter to be pruned
            :param string param_name: name of param
            :param int stage: the pruning stage
            :param boolean verbose: whether to print the pruning details

            :returns: mask for zeros

            :rtype: torch.ByteTensor
        """
        rule_id = -1
        for idx, r in enumerate(self.rule):
            m = re.match(r[0], param_name)
            if m is not None and len(param_name) == m.span()[1]:
                rule_id = idx
                break
        if rule_id > -1:
            sparsity = self.rule[rule_id][2][stage]
            fn_prune = self.rule[rule_id][-1]
            fn_importance = self.rule[rule_id][3]
            if verbose:
                print("{param_name:^30} | {stage:5d} | {spars:.3f}".
                      format(param_name=param_name, stage=stage, spars=sparsity))
            if fn_importance is None or fn_importance == 'default':
                mask = fn_prune(param=param, sparsity=sparsity)
            elif fn_importance == 'abs':
                mask = fn_prune(param=param, sparsity=sparsity, fn_importance=lambda x: x.abs())
            elif fn_importance == 'l1norm':
                mask = fn_prune(param=param, sparsity=sparsity, fn_importance=lambda x: x.norm(1, -1))
            elif fn_importance == 'l2norm':
                mask = fn_prune(param=param, sparsity=sparsity, fn_importance=lambda x: x.norm(2, -1))
            else:
                mask = fn_prune(param=param, sparsity=sparsity, fn_importance=fn_importance)
            return mask
        else:
            if verbose:
                print("{param_name:^30} | skipping".format(param_name=param_name))
            return None

    def prune(self, model, stage=0, update_masks=False, verbose=False):
        """ Prunes models
        
            :param torch.nn.Module model: model to be pruned
            :param int stage: the pruning stage {default: 0}
            :param boolean update_masks: whether to update masks
            :param boolean verbose: whether to print the pruning details
        """
        update_masks = True if update_masks or len(self.masks) == 0 else False
        if verbose:
            print("=" * 89)
            print("Pruning Models")
            if len(self.masks) == 0:
                print("Initializing Masks")
            elif update_masks:
                print("Updating Masks")
            print("=" * 89)
            print("{name:^30} | stage | sparsity".format(name='param_name'))
        for param_name, param in model.named_parameters():
            if 'AuxLogits' not in param_name:
                # deal with googlenet
                if param.dim() > 1:
                    if update_masks:
                        mask = self.prune_param(param=param.data, param_name=param_name,
                                                stage=stage, verbose=verbose)
                        if mask is not None:
                            self.masks[param_name] = mask
                    else:
                        if param_name in self.masks:
                            mask = self.masks[param_name]
                            param.data.masked_fill_(mask, 0)
       
        if verbose:
            print("=" * 89)
from torch.nn.modules import Module
import torch
from math import ceil

class Bay_Loss(Module):
    def __init__(self, use_background, device):
        """ Initializes a Bay_Loss object
        
            :param boolean use_background: whether the image background will be considered
            :param string device: device to be used for the computation of Bayesian Loss
        """
        super(Bay_Loss, self).__init__()
        self.device = device
        self.use_bg = use_background

    def forward(self, prob_list, target_list, pre_density):
        """ Implements the forward pass
        
            :param list prob_list: list of input images
            :param list target_list: list of ground truth density maps
            :param list pre_density: list of generated density maps

            :returns: loss value of the forward pass

            :rtype: double
        """ 
        loss = 0
        for idx, prob in enumerate(prob_list):  # iterative through each sample
            if prob is None:  # image contains no annotation points
                pre_count = torch.sum(pre_density[idx])
                target = torch.zeros((1,), dtype=torch.float32, device=self.device)
            else:
                N = len(prob)
                if self.use_bg:
                    target = torch.zeros((N,), dtype=torch.float32, device=self.device)
                    target[:-1] = target_list[idx]
                else:
                    # target = target_list[idx]
                    target = target_list[idx]
                pre_count = torch.sum(pre_density[idx].view((1, -1)) * prob, dim=1)  # flatten into vector

            res = torch.abs(target - pre_count)
            num = ceil(0.9 * (len(res) - 1))
            loss += torch.sum(torch.topk(res[:-1], num, largest=False)[0])
            loss += res[-1]
        loss = loss / len(prob_list)
        return loss
    
class Post_Prob(Module):
    def __init__(self, sigma, c_size, stride, background_ratio, use_background, device):
        """ Initializes a Post_Prob object
        
        Arguments:
            sigma {double} -- sigma value of the object
            c_size {int} -- crop size of the images
            stride {int} -- stride of the convolutional layer
            background_ratio {double} -- ratio of the image backgrounds
            use_background {boolean} -- whether the image background will be considered
            device {string} -- device to be used
        """
        super(Post_Prob, self).__init__()
        assert c_size % stride == 0

        self.sigma = sigma
        self.bg_ratio = background_ratio
        self.device = device
        # coordinate is same to image space, set to constant since crop size is same
        self.cood = torch.arange(0, c_size, step=stride,
                                 dtype=torch.float32, device=device) + stride / 2
        self.cood.unsqueeze_(0)
        self.softmax = torch.nn.Softmax(dim=0)
        self.use_bg = use_background

    def forward(self, points, st_sizes):
        """ Implements the forward pass
        
            :param list points: list of points in the input image density maps
            :param list st_sizes: minimum dimension size of each of the input images

            :returns: list of generated density maps

            :rtype: list
        """
        num_points_per_image = [len(points_per_image) for points_per_image in points]
        all_points = torch.cat(points, dim=0)

        if len(all_points) > 0:
            x = all_points[:, 0].unsqueeze_(1)
            y = all_points[:, 1].unsqueeze_(1)
            x_dis = -2 * torch.matmul(x, self.cood) + x * x + self.cood * self.cood
            y_dis = -2 * torch.matmul(y, self.cood) + y * y + self.cood * self.cood
            y_dis.unsqueeze_(2)
            x_dis.unsqueeze_(1)
            dis = y_dis + x_dis
            dis = dis.view((dis.size(0), -1))

            dis_list = torch.split(dis, num_points_per_image)
            prob_list = []
            for dis, st_size in zip(dis_list, st_sizes):
                if len(dis) > 0:
                    if self.use_bg:
                        min_dis = torch.clamp(torch.min(dis, dim=0, keepdim=True)[0], min=0.0)
                        d = st_size * self.bg_ratio
                        bg_dis = (d - torch.sqrt(min_dis))**2
                        dis = torch.cat([dis, bg_dis], 0)  # concatenate background distance to the last
                    dis = -dis / (2.0 * self.sigma ** 2)
                    prob = self.softmax(dis)
                else:
                    prob = None
                prob_list.append(prob)
        else:
            prob_list = []
            for _ in range(len(points)):
                prob_list.append(None)
        return prob_list
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
import scipy.io as scio
from utils.graphConstructionFromStandard import *

import numpy as np
import random
random.seed(0)

dataset_path = {'seed4': '/home/wsl/f/Common_Win_WSL_Datasets/SEEDIV_DATASET/eeg_feature_smooth', 'seed3' : '/home/wsl/f/Common_Win_WSL_Datasets/SEED_DATASET/ExtractedFeatures'}

def norminx(data, type='min_max'):
    '''
    description: norm in x dimension
    param {type}:
        data: array
    return {type}
    '''
    for i in range(data.shape[0]):
        data[i] = normalization(data[i], type)
    return data


def norminy(data, type='min_max'):
    dataT = data.T
    for i in range(dataT.shape[0]):
        dataT[i] = normalization(dataT[i], type)
    return dataT.T

def optimizer_scheduler(optimizer, init_lr, p):
    """
    Adjust the learning rate of optimizer
    :param optimizer: optimizer for updating parameters
    :param init_lr: initial learning rate.
    :param p: a variable for adjusting learning rate
    :return: optimizer
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr / (1. + 10 * p) ** 0.75

    return optimizer

def normalization(data, type = 'min_max'):
    '''
    description:
    param {type}: min_max, z_score
    return {type}
    '''
    if type == 'min_max':
        _range = np.max(data) - np.min(data)
        ret = (data - np.min(data)) / _range
    elif type == 'z_score':
        x_mean = np.mean(data)
        x_std = np.std(data)
        ret = (data - x_mean) / x_std

    return ret

# package the data and label into one class
class CustomDataset(Dataset):
    # initialization: data, label, and domain label
    def __init__(self, Data, Label, DomainLabel):
        self.Data = Data
        self.Label = Label
        self.DomainLabel = DomainLabel

        # shuffle datasets.
        # shuffle_ix = np.random.permutation(np.arange(len(Data)))
        # self.Data =  Data[shuffle_ix, :]
        # self.Label = Label[shuffle_ix, :]
        # self.DomainLabel = DomainLabel[shuffle_ix, :]

    # get the size of data
    def __len__(self):
        return len(self.Data)

    # get the data and label
    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])
        domain_label = torch.LongTensor(self.DomainLabel[index])
        return data, label, domain_label

class CustomSubDependentDataset(Dataset):
    def __init__(self, numpy_Data, numpy_Label):
        self.Data = numpy_Data
        self.Label = numpy_Label

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, index):
        data = torch.Tensor(self.Data[index])
        label = torch.LongTensor(self.Label[index])

        return data, label

def get_number_of_label_n_trial(dataset_name):
    '''
    description: get the number of categories, trial number and the corresponding labels
    param {type}
    return {type}:
        trial: int
        label: int
        label_xxx: list 3*15
    '''
    # global variables
    label_seed4 = [[1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
                   [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2,
                       0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
                   [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]]
    label_seed3 = [[2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0],
                   [2, 1, 0, 0, 1, 2, 0, 1, 2, 2, 1, 0, 1, 2, 0]]
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')

def reshape_data(data, label):
    '''
    description: reshape data and initiate corresponding label vectors
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*310
        reshape_label: array, x*1
    '''
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):  # 15 trials
        # one_data: one trial data shape: [62, 235, 5]
        one_data = np.reshape(np.transpose(
            data[i], (1, 2, 0)), (-1, 310), order='F')  # order F means that: coloum first,
        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label

def get_data_label_frommat(mat_path, dataset_name, session_id):
    _, _, labels = get_number_of_label_n_trial(dataset_name)
    mat_data = scio.loadmat(mat_path)
    # for i in range(1, 17):
    #     cur_name = 'de_LDS' + str(i)
    #     print(mat_data[cur_name].shape)
    # exit()
    mat_de_data = {key:value for key, value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())
    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    return one_sub_data, one_sub_label

def get_single_band_data_label_frommat(mat_path, dataset_name, session_id, band_id):
    _, _, labels = get_number_of_label_n_trial(dataset_name)
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key:value for key, value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values())
    one_sub_single_band_data, one_sub_single_band_label = reshape_single_band_data(mat_de_data, labels[session_id], band_id=band_id)
    return one_sub_single_band_data, one_sub_single_band_label

def reshape_single_band_data(data, label, band_id = 0):
    '''
    description: reshape data and initiate corresponding label vectors
    param {type}:
        data: list
        label: list
    return {type}:
        reshape_data: array, x*62
        reshape_label: array, x*1
    '''
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):  # 15 trials
        # one_data: one trial data shape: [62, 235, 5]
        one_data = np.transpose(data[i][:, :, band_id], (1, 0)) # [62, 235] -> [235, 62]
        one_label = np.full((one_data.shape[0], 1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label

def load_single_band_data(dataset_name, band_id):
    '''
    description: get all the data from one dataset
    param {type}
    return {type}:
        data: list 3(sessions) * 15(subjects), each data is x * 62
        label: list 3*15, x*1
    '''
    path, allmats = get_allmats_name(dataset_name)
    data = [([0] * 15) for i in range(3)]  # 定义list shape
    label = [([0] * 15) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = path + '/session' + str(i) + '/' + allmats[i][j]
            one_data, one_label = get_single_band_data_label_frommat(
                mat_path, dataset_name, i, band_id=band_id)
            data[i][j] = one_data.copy()
            label[i][j] = one_label.copy()
    return np.array(data), np.array(label)

def get_allmats_name(dataset_name):
    path = dataset_path[dataset_name]
    sessions = os.listdir(path)
    sessions.sort()
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)
    return path, allmats

def load_data(dataset_name):
    '''
    description: get all the data from one dataset
    param {type}
    return {type}:
        data: list 3(sessions) * 15(subjects), each data is x * 310
        label: list 3*15, x*1
    '''
    path, allmats = get_allmats_name(dataset_name)
    data = [([0] * 15) for i in range(3)]  # 定义list shape
    label = [([0] * 15) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])):
            mat_path = path + '/session' + str(i) + '/' + allmats[i][j]
            one_data, one_label = get_data_label_frommat(
                mat_path, dataset_name, i)
            data[i][j] = one_data.copy()
            label[i][j] = one_label.copy()
    return np.array(data), np.array(label)

def get_adj_from_standard():
    adj = format_adj_matrix_from_standard(SEED_CHANNEL_LIST, STANDARD_1005_CHANNEL_LOCATION_DICT)

    return adj


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def mmd(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


def mmd_rbf_accelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    loss = 0
    for i in range(batch_size):
        s1, s2 = i, (i+1) % batch_size
        t1, t2 = s1+batch_size, s2+batch_size
        loss += kernels[s1, s2] + kernels[t1, t2]
        loss -= kernels[s1, t2] + kernels[s2, t1]
    return loss / float(batch_size)


def mmd_linear(f_of_X, f_of_Y):
    delta = f_of_X - f_of_Y
    loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
    return loss


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps = 1e-3, max_iter = 1000, reduction='mean'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to('cuda')
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to('cuda')

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            # print(mu.device, self.M(C,u,v).device)
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

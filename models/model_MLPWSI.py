
import torch
import numpy as np 
from x_transformers import CrossAttender

import torch
import torch.nn as nn
from torch import nn
from einops import reduce

from x_transformers import Encoder
from torch.nn import ReLU

from models.layers.cross_attention import FeedForward, MMAttentionLayer
import torch.nn.functional as F
def exists(val):
    return val is not None


class MLPWSI(nn.Module):
    def __init__(self, hidden_dims=[512, 256, 128], activation=F.relu, **model_dict):
        super(MLPWSI, self).__init__()
        self.input_dim = model_dict['wsi_embedding_dim']  # 序列上的特征数
        self.output_dim = model_dict['wsi_projection_dim']
        self.num_classes = model_dict['n_classes']
        self.device = model_dict['device']  # 获取设备信息

        self.hidden_dims = hidden_dims
        self.activation = activation

        # 创建模型层
        layers = []
        layers.append(nn.Flatten(start_dim=1))  # 展平输入张量

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(self.input_dim * 4000, hidden_dims[0]))  # 假设序列长度为4000
        layers.append(nn.ReLU())

        # 后续隐藏层
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))
            layers.append(nn.ReLU())

        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], self.num_classes))

        self.layers = nn.Sequential(*layers).to(self.device)  # 将所有层移到指定设备

    def forward(self, x):
        x = x.to(self.device)  # 确保输入数据也在正确的设备上
        x = self.layers(x)
        return x
from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

"""
Implements DeepMISL for unimodal (WSI only) and multimodal (WSI + pathways). The combination of WSI and pathways can be done via concatenation
or bilinear fusion. 

Yao, Jiawen, et al. "Whole slide images based cancer survival prediction using attention guided deep multiple instance learning networks." Medical Image Analysis 65 (2020): 101789.

"""

################################
### Deep Sets Implementation ###
################################
class DeepMISL(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg = "small", dropout=0.25, df_comp=None, dim_per_path_1=16, dim_per_path_2=64, device="cpu", n_classes=4):
        r"""
        Deep Sets Implementation.

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(DeepMISL, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        self.phi = nn.Sequential(*[nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)])
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        self.df_comp = df_comp
        self.dim_per_path_1 = dim_per_path_1
        self.num_pathways = self.df_comp.shape[1]
        self.dim_per_path_2 = dim_per_path_2
        self.input_dim = omic_input_dim

        ### Constructing Genomic SNN
        if self.fusion != None:
            self.num_pathways = self.df_comp.shape[1]
            M_raw = torch.Tensor(self.df_comp.values)
            self.mask_1 = torch.repeat_interleave(M_raw, self.dim_per_path_1, dim=1)

            self.fc_1_weight = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(self.input_dim, self.dim_per_path_1*self.num_pathways)))
            self.fc_1_bias = nn.Parameter(torch.rand(self.dim_per_path_1*self.num_pathways))

            self.fc_2_weight = nn.init.xavier_normal_(nn.Parameter(torch.FloatTensor(self.dim_per_path_1*self.num_pathways, self.dim_per_path_2*self.num_pathways)))
            self.fc_2_bias = nn.Parameter(torch.rand(self.dim_per_path_2*self.num_pathways))

            self.mask_2 = np.zeros([self.dim_per_path_1*self.num_pathways, self.dim_per_path_2*self.num_pathways])
            for (row, col) in zip(range(0, self.dim_per_path_1*self.num_pathways, self.dim_per_path_1), range(0, self.dim_per_path_2*self.num_pathways, self.dim_per_path_2)):
                self.mask_2[row:row+self.dim_per_path_1, col:col+self.dim_per_path_2] = 1
            self.mask_2 = torch.Tensor(self.mask_2)

            self.upscale = nn.Sequential(
                nn.Linear(self.dim_per_path_2*self.num_pathways, int(256/4)),
                nn.ReLU(),
                nn.Linear(int(256/4), 256)
            )

            hidden = [256, 256]
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
        
            if self.fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)

        if self.fusion != None:
            self.device = device
            self.fc_1_weight.to(self.device)
            self.fc_1_bias.to(self.device)
            self.mask_1 = self.mask_1.to(self.device)

            self.fc_2_weight.to(self.device)
            self.fc_2_bias.to(self.device)
            self.mask_2 = self.mask_2.to(self.device)

            self.mm = self.mm.to(self.device)
            self.classifier = self.classifier.to(self.device)

            self.activation = nn.ReLU()


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.phi = nn.DataParallel(self.phi, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        x_path = kwargs['data_WSI']
        x_path = x_path.squeeze()

        h_path = self.phi(x_path).sum(axis=0)
        h_path = self.rho(h_path)

        if self.fusion is not None:
            x_omic = kwargs['data_omics']
            x_omic = x_omic.squeeze(0)

            out = torch.matmul(x_omic, self.fc_1_weight * self.mask_1) + self.fc_1_bias
            out = self.activation(out)
            out = torch.matmul(out, self.fc_2_weight * self.mask_2) + self.fc_2_bias 

            #---> apply linear transformation to upscale the dim_per_pathway (from 32 to 256) Lin, GELU, dropout, 
            h_omic = self.upscale(out)
            
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path # [256] vector

        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        # Y_hat = torch.topk(logits, 1, dim = 1)[1]
        # hazards = torch.sigmoid(logits)
        # S = torch.cumprod(1 - hazards, dim=1)
        
        # return hazards, S, Y_hat, None, None
        return logits



################################
# Attention MIL Implementation #
################################
class MIL_Attention_FC_surv(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, size_arg = "small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Attention_FC_surv, self).__init__()
        self.fusion = fusion
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}

        ### Deep Sets Architecture Construction
        size = self.size_dict_path[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Constructing Genomic SNN
        if self.fusion is not None:
            hidden = [256, 256]
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)
        
            if self.fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        x_path = kwargs['x_path']

        A, h_path = self.attention_net(x_path)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic)
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path # [256] vector

        logits  = self.classifier(h).unsqueeze(0) # logits needs to be a [1 x 4] vector 
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S, Y_hat, None, None



######################################
# Deep Attention MISL Implementation #
######################################
class MIL_Cluster_FC_surv(nn.Module):
    def __init__(self, omic_input_dim=None, fusion=None, num_clusters=10, size_arg = "small", dropout=0.25, n_classes=4):
        r"""
        Attention MIL Implementation

        Args:
            omic_input_dim (int): Dimension size of genomic features.
            fusion (str): Fusion method (Choices: concat, bilinear, or None)
            size_arg (str): Size of NN architecture (Choices: small or large)
            dropout (float): Dropout rate
            n_classes (int): Output shape of NN
        """
        super(MIL_Cluster_FC_surv, self).__init__()
        self.size_dict_path = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256]}
        self.num_clusters = num_clusters
        self.fusion = fusion
        
        ### FC Cluster layers + Pooling
        size = self.size_dict_path[size_arg]
        phis = []
        for phenotype_i in range(num_clusters):
            phi = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout),
                   nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
            phis.append(nn.Sequential(*phi))
        self.phis = nn.ModuleList(phis)
        self.pool1d = nn.AdaptiveAvgPool1d(1)
        
        ### WSI Attention MIL Construction
        fc = [nn.Linear(size[1], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.rho = nn.Sequential(*[nn.Linear(size[1], size[2]), nn.ReLU(), nn.Dropout(dropout)])

        ### Genomic SNN Construction + Multimodal Fusion
        if fusion is not None:
            hidden = self.size_dict_omic['small']
            fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            self.fc_omic = nn.Sequential(*fc_omic)

            if fusion == 'concat':
                self.mm = nn.Sequential(*[nn.Linear(size[2]*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
            elif self.fusion == 'bilinear':
                self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
            else:
                self.mm = None

        self.classifier = nn.Linear(size[2], n_classes)


    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() >= 1:
            device_ids = list(range(torch.cuda.device_count()))
            self.attention_net = nn.DataParallel(self.attention_net, device_ids=device_ids).to('cuda:0')
        else:
            self.attention_net = self.attention_net.to(device)

        if self.fusion is not None:
            self.fc_omic = self.fc_omic.to(device)
            self.mm = self.mm.to(device)

        self.phis = self.phis.to(device)
        self.pool1d = self.pool1d.to(device)
        self.rho = self.rho.to(device)
        self.classifier = self.classifier.to(device)


    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        cluster_id = kwargs['cluster_id'].detach().cpu().numpy()

        ### FC Cluster layers + Pooling
        h_cluster = []
        for i in range(self.num_clusters):
            h_cluster_i = self.phis[i](x_path[cluster_id==i])
            if h_cluster_i.shape[0] == 0:
                h_cluster_i = torch.zeros((1,512)).to(torch.device('cuda'))
            h_cluster.append(self.pool1d(h_cluster_i.T.unsqueeze(0)).squeeze(2))
        h_cluster = torch.stack(h_cluster, dim=1).squeeze(0)

        ### Attention MIL
        A, h_path = self.attention_net(h_cluster)  
        A = torch.transpose(A, 1, 0)
        A_raw = A 
        A = F.softmax(A, dim=1) 
        h_path = torch.mm(A, h_path)
        h_path = self.rho(h_path).squeeze()

        ### Attention MIL + Genomic Fusion
        if self.fusion is not None:
            x_omic = kwargs['x_omic']
            h_omic = self.fc_omic(x_omic)
            if self.fusion == 'bilinear':
                h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
            elif self.fusion == 'concat':
                h = self.mm(torch.cat([h_path, h_omic], axis=0))
        else:
            h = h_path

        logits  = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        return hazards, S, Y_hat, None, None
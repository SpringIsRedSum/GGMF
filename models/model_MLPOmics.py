import torch.nn as nn
from torch.nn import ReLU, ELU
import torch

"""

Implement a MLP to handle tabular omics data 

"""

class MLPOmics(nn.Module):
    def __init__(
        self, 
        input_dim,
        n_classes=4, 
        projection_dim = 512, 
        dropout = 0.1, 
        ):
        super(MLPOmics, self).__init__()
        
        # self
        self.projection_dim = projection_dim
        self.layer = None
        self.input_dim = input_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim//2), ReLU(), nn.Dropout(dropout),
            nn.Linear(projection_dim//2, projection_dim//2), ReLU(), nn.Dropout(dropout)
        ) 

        self.to_logits = nn.Sequential(
                nn.Linear(projection_dim//2, n_classes)
            )

    def forward(self, **kwargs):
        try:
            self.cuda()

            #---> unpack

            data_omics = [kwargs['x_omic%d' % i] for i in range(1, len(kwargs)+1)]
            # 将列表里的所有张量堆叠 在一起
            data_omics_bag = torch.stack(data_omics, dim=0)

            # data_omics = kwargs["data_omics"].float().cuda().squeeze()

            if(self.layer == None):
                self.layer = nn.Linear(data_omics_bag.shape[1],self.input_dim).to(data_omics_bag.device)
            data = self.layer(data_omics_bag)
            #---> project omics data to projection_dim/2
            data = self.net(data) #[B, n]

            #---->predict
            logits = self.to_logits(data) #[B, n_classes]
        except BaseException as e:
            import pdb
            print("检查 waargs的内容")
            pdb.set_trace()

        return logits
    
    def captum(self, omics):

        self.cuda()

        #---> unpack
        data_omics = omics.float().cuda().squeeze()
        
        #---> project omics data to projection_dim/2
        data = self.net(data_omics) #[B, n]

        #---->predict
        logits = self.to_logits(data) #[B, n_classes]

        #---> get risk 
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1 - hazards, dim=1)
        risk = -torch.sum(survival, dim=1)

        #---> return risk 
        return risk




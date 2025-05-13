import numpy as np
import torch
import math
from torch import nn

from sklearn.preprocessing import RobustScaler

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)

    args:
        dim1 (int): Dimension of input features
        dim2 (int): Dimension of output features
        dropout (float): Dropout rate
    """
    import torch.nn as nn

    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))



class FeedForward(nn.Module):
    def __init__(self, dim, mult=1, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(self.norm(x))



class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # Einsum does matrix multiplication for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just a way to do batch matrix multiplication
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, 8)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
        )

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        forward = self.feed_forward(x)
        out = self.norm2(forward + x)
        return out







class SurvPathMamba(nn.Module):
    def __init__(self,
                omic_sizes=[100, 200, 300, 400, 500, 600],
                wsi_embedding_dim=1024,
                dropout=0.1,
                num_classes=4,
                wsi_projection_dim=256,
                omic_names=[],
                **kwargs):
        super(SurvPathMamba, self).__init__()
        # --------> general props
        self.num_patches = kwargs["num_patches"]


        self.num_pathways = len(omic_sizes)
        self.dropout = dropout

        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim

        self.changeChannel = None
        self.sumChannel = self.closest_lower_multiple_of_64(self.num_patches + self.num_pathways)
        # self.sumChannel是我希望得到的通道数 也就是 64的倍数

        if(self.num_patches + self.num_pathways ) != self.sumChannel:# 说明要修改

            self.changeChannel = self.num_patches + self.num_pathways  - self.sumChannel
            # 需要丢弃 这么多通道 self.changeChannel

        # ---------> omics preprocessing

        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            self.all_gene_names = all_gene_names # 每一行是 一种 基因

        # ----------> wso props 这两个维度需要搞清楚是什么?
        self.wsi_embedding_dim = wsi_embedding_dim  #这个应该是 每一张svs 维度是 1024
        self.wsi_projection = wsi_projection_dim # 这个是最后 模型需要的维度 256

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )  # 图片 的输入 维度 从 1024 -> 256

        # ---> omics props
        self.init_per_path_model(omic_sizes)

        # ------> cross attention props
        self.identity = nn.Identity()
        # 创建一个不改变输入数据的层，使得数据可以直接通过这个层

        # 输入这一层的形状是 torch.Size([batch, 4331, 256])
        # 我希望输出的形状是 torch.Size([batch, 4331, 128])

        # self.bn = TransformerBlock(wsi_projection_dim)


        from mambaModel.mambaBlock import NdMamba2_1d  # 输入 64的倍数 输出
        self.mambaBlock = NdMamba2_1d(self.sumChannel,self.sumChannel,1024,)

        # ----------------->logits props
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim , dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim)
        # 上面两层 不改变维度

        # 下面的维度变化 还要继续测试
        self.to_logits = nn.Sequential(
            nn.Linear(self.wsi_projection_dim * 2, int(self.wsi_projection_dim / 4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim / 4), self.num_classes)
        )
        self.robust_scaler = RobustScaler()

    def get_min_max_integer(self,number):
        if number >= 0:
            # 对于非负数，使用 math.ceil() 获取大于等于它的最小整数
            return math.ceil(number)
        else:
            # 对于负数，使用 math.floor() 获取小于等于它的最大整数
            return math.floor(number)

    def forward(self,**kwargs):

        wsi = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]

        # --------------> 获取 路径的 embdedings

        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(0) # 每个基因 向前传播后 最后堆叠在一起

        wsi_embed = self.wsi_projection_net(wsi) #wsi 传递给 self.wsi_projection_net 向前传播
        # 这里的维度 应该是 (4000 ,1024)
        # 对wsi 数据归一化标准化

        upBound = self.get_min_max_integer(h_omic_bag.max())
        lowBound = self.get_min_max_integer(h_omic_bag.min())
        Bound = max(abs(upBound),abs(lowBound))

        min_val = wsi_embed.min()
        max_val = wsi_embed.max()

        wsi_embed = (wsi_embed - min_val) / (max_val - min_val)
        wsi_embed = Bound * wsi_embed - Bound + Bound


        #对数据 四份
        x = wsi_embed.clone().to('cpu')

        # 将 PyTorch 张量转换为 NumPy 数组
        x_numpy = x.detach().numpy().squeeze()  # 使用 .squeeze() 移除长度为1的维度

        # 创建 RobustScaler 实例
        robust_scaler = RobustScaler()

        # 对 NumPy 数组进行拟合并转换回 PyTorch 张量
        x_scaled = torch.tensor(robust_scaler.fit_transform(x_numpy), dtype=torch.float32)
        x_scaled = x_scaled.unsqueeze(0)
        # 确保转换后的张量与原始张量具有相同的形状

        assert x_scaled.shape == wsi_embed.shape
        wsi_embed = x_scaled.to(wsi_embed.device)


        # print("218行 归一化后的数据源",wsi_embed)
        if self.changeChannel != None:
            #说明要丢弃 self.changeChannel 个通道
            #从 [0,kwargs"num_patches"] 中选择  self.changeChannel 个通道丢弃
            wsi_embed, self.dropped_channels = self.drop_random_channels(wsi_embed,self.changeChannel)
            # self.dropped_channels 记录了丢弃的通道数

        #应该在 cat之前 对不同的信息进行分类处理


        #将 wsi_映射到 制定范围的维度
        # print("208行",wsi_embed)
        # exit(209)
        # wsi_embed = self.mambaBlock(wsi_embed)
        # 这里对 基因组数据怎么处理呢?????????????????????
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        # 传递给self.cross_attender的tokens -> torch.Size([1, 能被64整除, 256])

        # print("输入给mambaBlock的形状",tokens.shape) #输入给mambaBlock的形状 torch.Size([1, 4288, 256])

        # 最开始给mamba的数据 量纲就不一样 慢慢检查数据维度 看看怎么融合吧！！！！！！！
        # print("传递给mamba的数据 ",tokens)
        # 还是这个模型出的问题 出来后全是         nan
        tokens = self.mambaBlock(tokens) # 提取出这个模型最关键的几个步骤
        # print("220行",tokens)

        mm_embed = self.identity(tokens)


        # mm_embed = self.cross_attender(tokens) # 这里需要一个结合层


        mm_embed = mm_embed.squeeze(0)

        # ------------> 向前传播 和 正则
        mm_embed = self.feed_forward(mm_embed)  # 不改变维度 expect [ _ ,128]
        mm_embed = self.layer_norm(mm_embed)  # 不改变维度


        mm_embed = mm_embed.unsqueeze(0)

        # ----------->
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]  # 选取 基因的维度
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)  # (1,1,256)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]  # 选取了 wsi 特征维度 (1,4000,256)
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)  # (1,1,256) 对第二个维度的数据取平均

        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)  # (1,2,256)


        logits = self.to_logits(embedding)  # 分类



        return logits

    def drop_random_channels(self,tensor, n):
        # 获取第二维的大小
        num_channels = tensor.shape[1]

        # 确保n不大于第二维的大小
        n = min(n, num_channels)

        # 生成一个随机排列的索引
        perm = torch.randperm(num_channels)

        # 选择要保留的通道索引
        keep_indices = perm[n:]

        # 使用索引来选择要保留的通道
        new_tensor = tensor[:, keep_indices, :]

        # 记录下被丢弃的通道索引
        dropped_channels = perm[:n]

        return new_tensor, dropped_channels

    def closest_lower_multiple_of_64(self,n):
        # 将n除以64，得到商和余数
        quotient, remainder = divmod(n, 64)

        # 如果余数为0，直接返回n
        if remainder == 0:
            return n

        # 如果余数不为0，返回商乘以64
        else:
            return quotient * 64

    def init_per_path_model(self, omic_sizes):
        hidden = [256, 256]  # 这里 硬编码 设置了 256 所以最后的维度对齐了
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

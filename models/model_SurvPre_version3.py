import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple, Optional


class MultiHeadGeneAttention(nn.Module):
    """多头自注意力机制用于基因选择

    参数:
        dim (int): 输入特征维度
        num_heads (int): 注意力头数量
        dropout (float): dropout比率
        bias (bool): 是否使用偏置
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 12,
                 dropout: float = 0.1,
                 bias: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim必须能被num_heads整除"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度
        self.scale = self.head_dim ** -0.5  # 注意力分数的缩放因子

        # Q,K,V的联合投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=bias)

        # 输出投影层
        self.proj = nn.Linear(dim, dim)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

        # 层归一化
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        参数:
            x: 输入张量, shape [batch_size, num_genes, dim]

        返回:
            attended_features: 注意力加权后的特征
            importance_scores: 基因重要性分数
            attention_weights: 注意力权重
        """
        B, N, C = x.shape  # batch_size, num_genes, dim

        # 1. 应用层归一化
        x = self.norm(x)

        # 2. 生成Q,K,V并分头
        qkv = self.qkv(x)  # [B, N, 3*C]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 3. 计算注意力分数
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        # 4. 注意力加权
        x = (attn @ v).transpose(1, 2)  # [B, N, num_heads, head_dim]
        x = x.reshape(B, N, C)  # [B, N, dim]
        x = self.proj(x)  # [B, N, dim]

        # 5. 计算基因重要性分数（所有头的注意力权重之和）
        importance = attn.sum(dim=1).mean(dim=1)  # [B, N]

        return x, importance, attn


def SNN_Block(dim1: int, dim2: int, dropout: float = 0.25) -> nn.Sequential:
    """Self-Normalizing Neural Network Block

    参数:
        dim1: 输入维度
        dim2: 输出维度
        dropout: Dropout比率
    """
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False)
    )


class SurvPath(nn.Module):
    """

    参数:
        omic_sizes (List[int]): 每个组学数据的特征维度
        wsi_embedding_dim (int): WSI特征的嵌入维度
        dropout (float): dropout比率
        num_classes (int): 分类类别数
        wsi_projection_dim (int): WSI投影维度
        topk_ratio (float): 选择的基因比例
        num_heads (int): 注意力头数量
    """

    def __init__(
            self,
            omic_sizes: List[int] = [100, 200, 300, 400, 500, 600],
            wsi_embedding_dim: int = 1024,
            dropout: float = 0.1,
            num_classes: int = 4,
            wsi_projection_dim: int = 256,
            topk_ratio: float = 0.8,
            num_heads: int = 12 # 设置交叉注意力的 头
    ):
        super().__init__()

        # 基本属性
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim
        self.topk_ratio = topk_ratio
        self.num_heads = num_heads

        # 计算总基因数
        self.total_genes = sum(omic_sizes)

        # 初始化基因重要性跟踪
        self.gene_importance = nn.Parameter(torch.zeros(self.total_genes), requires_grad=False)
        self.gene_importance_count = nn.Parameter(torch.zeros(1), requires_grad=False)

        # WSI特征投影网络
        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 基因组特征处理网络
        self.init_per_path_model(omic_sizes)

        # 多头自注意力用于基因选择
        self.gene_attention = MultiHeadGeneAttention(
            dim=wsi_projection_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=wsi_projection_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 双线性融合层
        self.bilinear_fusion = nn.Bilinear(
            wsi_projection_dim,
            wsi_projection_dim,
            wsi_projection_dim
        )

        # 特征融合网络 - 注意输入维度是wsi_projection_dim
        self.fusion_net = nn.Sequential(
            nn.Linear(wsi_projection_dim, wsi_projection_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(wsi_projection_dim, wsi_projection_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # 层归一化
        self.norm1 = nn.LayerNorm(wsi_projection_dim)
        self.norm2 = nn.LayerNorm(wsi_projection_dim)

        # 分类器
        self.classifier = nn.Linear(wsi_projection_dim, num_classes)

    def init_per_path_model(self, omic_sizes: List[int]) -> None:
        """初始化每个组学路径的处理网络

        参数:
            omic_sizes: 每个组学数据的特征维度列表
        """
        hidden = [256, 256]  # 隐藏层维度
        sig_networks = []

        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(input_dim, hidden[0])]
            for i in range(len(hidden) - 1):
                fc_omic.append(SNN_Block(hidden[i], hidden[i + 1]))
            sig_networks.append(nn.Sequential(*fc_omic))

        self.sig_networks = nn.ModuleList(sig_networks)

    def select_genes(self, h_omic_bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用多头自注意力机制选择重要的基因

        参数:
            h_omic_bag: 基因特征张量 [batch_size, num_genes, hidden_dim]

        返回:
            selected_features: 选中的基因特征
            attention_weights: 注意力权重
            importance_scores: 重要性分数
        """
        batch_size, num_genes, _ = h_omic_bag.shape

        # 1. 应用多头自注意力
        attended_features, importance_scores, attention_weights = self.gene_attention(h_omic_bag)

        # 2. 选择top k个基因
        k = int(num_genes * self.topk_ratio)
        topk_values, topk_indices = torch.topk(importance_scores, k, dim=1)

        # 3. 获取选中的基因特征
        selected_features = torch.stack([
            attended_features[i, topk_indices[i]] for i in range(batch_size)
        ])

        return selected_features, attention_weights, importance_scores

    def forward(self, **kwargs):
        """前向传播

        参数:
            kwargs: 包含以下键值对
                x_path: WSI特征 [batch_size, n_patches, embedding_dim]
                x_omic1, x_omic2, ...: 组学特征
                return_attn: 是否返回注意力权重

        返回:
            logits: 分类logits
            或 (logits, importance_scores, gene_attention, cross_attention)
        """

        wsi = kwargs['x_path']
        x_omic = [kwargs[f'x_omic{i}'] for i in range(1, self.num_pathways + 1)]
        return_attn = kwargs.get('return_attn', False)



        # 1. WSI特征投影
        wsi_embed = self.wsi_projection_net(wsi)  # [batch_size, n_patches, projection_dim]

        # 2. 处理组学特征
        h_omic = [net(sig_feat.float()) for net, sig_feat in zip(self.sig_networks, x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)  # [1, num_pathways, hidden_dim]

        # 3. 基因选择
        selected_genes, gene_attention, importance_scores = self.select_genes(h_omic_bag)
        # 更新基因重要性统计
        #with torch.no_grad():
          #  batch_importance = importance_scores.mean(dim=0)  # [total_genes]
            #self.gene_importance.data += batch_importance
           # self.gene_importance_count.data += 1

        # # 如果提供了基因名称，则分析并保存最重要的10个基因
        #
        # self.gene_names = kwargs.get('gene_names', None)
        #
        # avg_importance = self.gene_importance / (self.gene_importance_count + 1e-6)
        # topk_values, topk_indices = torch.topk(avg_importance, k=min(10, len(self.gene_names)))
        # top_genes_info = {
        #     'genes': [self.gene_names[i] for i in topk_indices],
        #     'importance_scores': topk_values.tolist()
        # }
        # print("\nTop 10 Important Genes:")
        # for gene, importance in zip(top_genes_info['genes'], top_genes_info['importance_scores']):
        #     print(f"{gene}: {importance:.4f}")
        #
        # print("importance_scores.shape:", importance_scores.shape)
        # print("gene_importance.shape:", self.gene_importance.shape)
        #
        # if torch.any(self.gene_importance != 0):
        #     print("[INFO] gene importance updated:", self.gene_importance[:10])
        # else:
        #     print("[WARN] gene importance still all zeros.")

        # 4. 准备交叉注意力输入
        selected_genes = selected_genes.permute(1, 0, 2)  # [seq_len, batch, dim]
        wsi_embed = wsi_embed.permute(1, 0, 2)

        # 5. 应用交叉注意力
        attended_wsi, cross_attention = self.cross_attention(
            query=selected_genes,
            key=wsi_embed,
            value=wsi_embed
        )

        # 6. 恢复维度顺序并应用残差连接
        attended_wsi = attended_wsi.permute(1, 0, 2)  # [batch, seq_len, dim]
        wsi_embed = wsi_embed.permute(1, 0, 2)

        # 应用第一个残差连接
        attended_wsi = self.norm1(attended_wsi + wsi_embed)  # 第一个残差连接

        # 7. 特征聚合
        attended_wsi_pooled = torch.mean(attended_wsi, dim=1)  # [batch, dim]
        gene_features = torch.mean(selected_genes.permute(1, 0, 2), dim=1)  # [batch, dim]

        # 8. 双线性特征融合

        # 应用双线性融合
        bilinear_features = self.bilinear_fusion(attended_wsi_pooled, gene_features)

        # 非线性变换
        fused_features = self.fusion_net(bilinear_features)

        # 9. 最终的残差连接和归一化
        final_features = self.norm2(fused_features + attended_wsi_pooled)  # 第二个残差连接

        # 10. 分类
        logits = self.classifier(final_features)

        if return_attn:
            return logits, importance_scores, gene_attention, cross_attention

        return logits

    def analyze_gene_importance(self, gene_names: List[str], importance_scores: torch.Tensor) -> List[dict]:
        """分析基因重要性

        参数:
            gene_names: 基因名称列表
            importance_scores: 重要性分数 [batch_size, num_genes]

        返回:
            sorted_genes: 按重要性排序的基因列表
        """
        avg_importance = importance_scores.mean(dim=0).cpu().detach().numpy()
        sorted_indices = np.argsort(-avg_importance)

        return [
            {'gene': gene_names[idx], 'importance': float(avg_importance[idx])}
            for idx in sorted_indices
        ]

    def visualize_attention(self, gene_names: List[str], attention_weights: torch.Tensor,
                            save_path: Optional[str] = None) -> None:
        """可视化注意力权重

        参数:
            gene_names: 基因名称列表
            attention_weights: 注意力权重
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 计算平均注意力权重
        avg_attn = attention_weights.mean(dim=1).mean(dim=0).cpu().detach().numpy()

        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_attn, xticklabels=gene_names, yticklabels=gene_names)
        plt.title('Gene-Gene Attention Pattern')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()  # 特征融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(wsi_projection_dim * 2, wsi_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(wsi_projection_dim, wsi_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 分类器
        self.classifier = nn.Linear(wsi_projection_dim, num_classes)

        # 层归一化
        self.norm1 = nn.LayerNorm(wsi_projection_dim)
        self.norm2 = nn.LayerNorm(wsi_projection_dim)

    def init_per_path_model(self, omic_sizes: List[int]) -> None:
        """初始化每个组学路径的处理网络

        参数:
            omic_sizes: 每个组学数据的特征维度列表
        """
        hidden = [256, 256]  # 隐藏层维度
        sig_networks = []

        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(input_dim, hidden[0])]
            for i in range(len(hidden) - 1):
                fc_omic.append(SNN_Block(hidden[i], hidden[i + 1]))
            sig_networks.append(nn.Sequential(*fc_omic))

        self.sig_networks = nn.ModuleList(sig_networks)

    def select_genes(self, h_omic_bag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """使用多头自注意力机制选择重要的基因

        参数:
            h_omic_bag: 基因特征张量 [batch_size, num_genes, hidden_dim]

        返回:
            selected_features: 选中的基因特征
            attention_weights: 注意力权重
            importance_scores: 重要性分数
        """
        batch_size, num_genes, _ = h_omic_bag.shape

        # 1. 应用多头自注意力
        attended_features, importance_scores, attention_weights = self.gene_attention(h_omic_bag)

        # 2. 选择top k个基因
        k = int(num_genes * self.topk_ratio)
        topk_values, topk_indices = torch.topk(importance_scores, k, dim=1)

        # 3. 获取选中的基因特征
        selected_features = torch.stack([
            attended_features[i, topk_indices[i]] for i in range(batch_size)
        ])

        return selected_features, attention_weights, importance_scores

    def forward(self, **kwargs):
        """前向传播

        参数:
            kwargs: 包含以下键值对
                x_path: WSI特征 [batch_size, n_patches, embedding_dim]
                x_omic1, x_omic2, ...: 组学特征
                return_attn: 是否返回注意力权重
        """
        wsi = kwargs['x_path']
        x_omic = [kwargs[f'x_omic{i}'] for i in range(1, self.num_pathways + 1)]
        return_attn = kwargs.get('return_attn', False)

        # 1. WSI特征投影
        wsi_embed = self.wsi_projection_net(wsi)  # [batch_size, n_patches, projection_dim]

        # 2. 处理组学特征
        h_omic = [net(sig_feat.float()) for net, sig_feat in zip(self.sig_networks, x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)  # [1, num_pathways, hidden_dim]

        # 3. 基因选择
        selected_genes, gene_attention, importance_scores = self.select_genes(h_omic_bag)


        # 4. 准备交叉注意力输入
        selected_genes = selected_genes.permute(1, 0, 2)  # [seq_len, batch, dim]
        wsi_embed = wsi_embed.permute(1, 0, 2)
        # print("Before attention:")
        # print(f"selected_genes shape: {selected_genes.shape}")
        # print(f"wsi_embed shape: {wsi_embed.shape}")

        # 5. 应用交叉注意力
        attended_wsi, cross_attention = self.cross_attention(
            query=selected_genes,
            key=wsi_embed,
            value=wsi_embed
        )

        # 6. 恢复维度顺序并应用残差连接
        attended_wsi = attended_wsi.permute(1, 0, 2)  # [batch, seq_len, dim]
        wsi_embed = wsi_embed.permute(1, 0, 2)

        # 使用自适应池化调整 wsi_embed 的维度
        wsi_embed = F.adaptive_avg_pool2d(
            wsi_embed.permute(0, 2, 1),  # [batch, dim, seq_len]
            (wsi_embed.size(2), attended_wsi.size(1))  # 保持特征维度不变，调整序列长度
        ).permute(0, 2, 1)  # 变回 [batch, seq_len, dim]

        # 应用第一个残差连接
        attended_wsi = self.norm1(attended_wsi + wsi_embed)

        # 7. 特征聚合
        attended_wsi_pooled = torch.mean(attended_wsi, dim=1)  # [batch, dim]
        gene_features = torch.mean(selected_genes.permute(1, 0, 2), dim=1)  # [batch, dim]

        # 8. 双线性特征融合
        bilinear_features = self.bilinear_fusion(attended_wsi_pooled, gene_features)  # [batch, dim]

        # 9. 特征融合网络
        fused_features = self.fusion_net(bilinear_features)  # [batch, dim]

        # 10. 最终的残差连接和归一化
        final_features = self.norm2(fused_features + attended_wsi_pooled)  # [batch, dim]

        # 11. 分类
        logits = self.classifier(final_features)

        if return_attn:
            return logits, importance_scores, gene_attention, cross_attention
        return logits

    def analyze_gene_importance(self, gene_names: List[str], importance_scores: torch.Tensor) -> List[dict]:
        """分析基因重要性

        参数:
            gene_names: 基因名称列表
            importance_scores: 重要性分数 [batch_size, num_genes]

        返回:
            sorted_genes: 按重要性排序的基因列表
        """
        avg_importance = importance_scores.mean(dim=0).cpu().detach().numpy()
        sorted_indices = np.argsort(-avg_importance)

        return [
            {'gene': gene_names[idx], 'importance': float(avg_importance[idx])}
            for idx in sorted_indices
        ]

    def visualize_attention(self, gene_names: List[str], attention_weights: torch.Tensor,
                            save_path: Optional[str] = None) -> None:
        """可视化注意力权重

        参数:
            gene_names: 基因名称列表
            attention_weights: 注意力权重
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        # 计算平均注意力权重
        avg_attn = attention_weights.mean(dim=1).mean(dim=0).cpu().detach().numpy()

        # 创建热力图
        plt.figure(figsize=(12, 10))
        sns.heatmap(avg_attn, xticklabels=gene_names, yticklabels=gene_names)
        plt.title('Gene-Gene Attention Pattern')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        plt.show()
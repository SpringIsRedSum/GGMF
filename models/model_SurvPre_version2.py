import torch
import torch.nn as nn
import  torch.nn.functional as F
import  numpy as np
from einops import reduce


def exists(val):
    return val is not None


def SNN_Block(dim1, dim2, dropout=0.25):
    """Self-Normalizing Neural Network Block"""
    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False))


class SurvPath(nn.Module):
    def __init__(
            self,
            omic_sizes=[100, 200, 300, 400, 500, 600],
            wsi_embedding_dim=1024,
            dropout=0.1,
            num_classes=4,
            wsi_projection_dim=256,
            omic_names=[],
            topk_ratio=0.5,
            n_heads=8
    ):
        super(SurvPath, self).__init__()
        # 基本属性
        self.num_pathways = len(omic_sizes)
        self.dropout = dropout
        self.wsi_embedding_dim = wsi_embedding_dim
        self.wsi_projection_dim = wsi_projection_dim
        self.topk_ratio = topk_ratio
        # WSI特征投影网络
        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        # 基因组特征处理网络
        self.init_per_path_model(omic_sizes)
        # 基因选择的注意力层
        self.gene_attention = nn.Sequential(
            nn.Linear(wsi_projection_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        # 多头注意力层
        self.gene_guided_attention = nn.MultiheadAttention(
            embed_dim=wsi_projection_dim,
            num_heads=n_heads,
            dropout=dropout
        )

        # 注意力后的归一化层
        self.attention_norm = nn.LayerNorm(wsi_projection_dim)

        # 双线性特征融合层
        self.bilinear_fusion = nn.Sequential(
            nn.Linear(wsi_projection_dim * 2, wsi_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(wsi_projection_dim, wsi_projection_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 融合后的归一化层
        self.fusion_norm = nn.LayerNorm(wsi_projection_dim)
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(wsi_projection_dim, num_classes)
        )

    def init_per_path_model(self, omic_sizes):
        """初始化每个基因通路的处理网络"""
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

    def select_genes(self, h_omic_bag):
        """使用注意力机制选择重要的基因"""
        batch_size, num_genes, hidden_dim = h_omic_bag.shape

        # 计算注意力分数
        attention_scores = self.gene_attention(h_omic_bag)
        attention_scores = attention_scores.squeeze(-1)

        # 使用softmax获得注意力权重
        attention_weights = torch.softmax(attention_scores, dim=1)

        # 选择top k个基因
        k = int(num_genes * self.topk_ratio)

        topk_values, topk_indices = torch.topk(attention_weights, k, dim=1)

        # 创建mask
        mask = torch.zeros_like(attention_weights)
        for i in range(batch_size):
            mask[i].scatter_(0, topk_indices[i], 1.0)

        # 应用mask到特征
        selected_features = h_omic_bag * mask.unsqueeze(-1)

        # 只保留被选中的基因
        selected_features = torch.stack([
            selected_features[i, topk_indices[i]] for i in range(batch_size)
        ])

        return selected_features, attention_weights

    def forward(self, **kwargs):
        wsi = kwargs['x_path']  # [batch_size, n_patches, embedding_dim]
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, self.num_pathways + 1)]
        return_attn = kwargs.get("return_attn", False)

        # 1. 获取基因特征
        h_omic = [self.sig_networks[idx].forward(sig_feat.float())
                  for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic).unsqueeze(0)  # [1, n_genes, embedding_dim]

        # 2. 选择重要基因
        selected_genes, gene_attention = self.select_genes(h_omic_bag)  # [batch_size, k_genes, embedding_dim]

        # 3. WSI特征投影
        wsi_embed = self.wsi_projection_net(wsi)  # [batch_size, n_patches, projection_dim]
        wsi_embed_orig = wsi_embed  # 保存原始维度用于残差连接

        # 4. 准备多头注意力的输入
        selected_genes = selected_genes.permute(1, 0, 2)  # [k_genes, batch_size, projection_dim]
        wsi_embed = wsi_embed.permute(1, 0, 2)  # [n_patches, batch_size, projection_dim]

        # 5. 应用基因引导的多头注意力
        attended_wsi, attention_weights = self.gene_guided_attention(
            query=selected_genes,
            key=wsi_embed,
            value=wsi_embed
        )  # attended_wsi: [k_genes, batch_size, projection_dim]

        # 6. 调整维度并应用残差连接
        attended_wsi = attended_wsi.permute(1, 0, 2)  # [batch_size, k_genes, projection_dim]

        # 使用自适应池化调整 wsi_embed_orig 的序列长度
        # 首先调整维度以适应池化操作
        wsi_embed_resized = wsi_embed_orig.permute(0, 2, 1)  # [batch_size, projection_dim, n_patches]
        wsi_embed_resized = wsi_embed_resized.unsqueeze(2)  # [batch_size, projection_dim, 1, n_patches]

        # 应用自适应池化
        wsi_embed_resized = F.adaptive_avg_pool2d(
            wsi_embed_resized,
            (1, attended_wsi.size(1))  # 目标大小：[1, k_genes]
        )  # [batch_size, projection_dim, 1, k_genes]

        # 调整回原始维度顺序
        wsi_embed_resized = wsi_embed_resized.squeeze(2)  # [batch_size, projection_dim, k_genes]
        wsi_embed_resized = wsi_embed_resized.permute(0, 2, 1)  # [batch_size, k_genes, projection_dim]

        # 应用残差连接和归一化
        attended_wsi = self.attention_norm(attended_wsi + wsi_embed_resized)

        # 7. 特征聚合
        attended_wsi = torch.mean(attended_wsi, dim=1)  # [batch_size, projection_dim]
        gene_features = torch.mean(selected_genes.permute(1, 0, 2), dim=1)  # [batch_size, projection_dim]

        # 8. 双线性特征融合
        combined_features = torch.cat([attended_wsi, gene_features], dim=1)  # [batch_size, projection_dim*2]
        fused_features = self.bilinear_fusion(combined_features)  # [batch_size, projection_dim]

        # 9. 最终的残差连接和归一化
        fused_features = self.fusion_norm(fused_features + attended_wsi)  # [batch_size, projection_dim]

        # 10. 最终分类
        logits = self.classifier(fused_features)  # [batch_size, num_classes]

        # 添加调试信息
        if kwargs.get('debug', False):
            print("Debug dimensions:")
            print(f"WSI input shape: {wsi.shape}")
            print(f"Selected genes shape: {selected_genes.shape}")
            print(f"Original WSI embed shape: {wsi_embed_orig.shape}")
            print(f"Attended WSI shape before resize: {attended_wsi.shape}")
            print(f"WSI embed after resize shape: {wsi_embed_resized.shape}")
            print(f"Final fused features shape: {fused_features.shape}")

        if return_attn:
            return logits, gene_attention, attention_weights
        return logits

    def analyze_attention_patterns(self, attention_weights, gene_names=None):
        """分析多头注意力的模式"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        n_heads = attention_weights.shape[0]
        fig, axes = plt.subplots(2, n_heads // 2, figsize=(20, 10))
        axes = axes.ravel()

        for head in range(n_heads):
            attn = attention_weights[head].detach().cpu().numpy()
            sns.heatmap(attn, ax=axes[head], cmap='viridis')
            axes[head].set_title(f'Head {head + 1}')
            if gene_names is not None:
                axes[head].set_yticklabels(gene_names, rotation=0)

        plt.tight_layout()
        return fig
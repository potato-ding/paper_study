# ViT精读



### 灵感：

- 将图片不依赖CNN，而是用标准的Transformer就能表现得很好
- 归纳偏置的博弈
  - CNN：有很强的归纳偏置（平移等变性和局部性）意味着CNN天生就知道猫在左上角还是右上角以及像素点和周围像素点关系更密切
  - transformer缺乏这些偏置，将图像看作一堆patches，甚至不知道patches之间的位置关系
  - 数据量不够（中等规模数据集如 ImageNet），ViT 打不过 ResNet 。但是！**一旦数据量巨大 (14M-300M 张图)，大规模训练可以“战胜”归纳偏置 (Large scale training trumps inductive bias)** 

### 如何把图片转换成一句话，喂给transformer

- 切块：将图切成小方块

假设图片大小是 $224 \times 224$，设 Patch 大小 ($P$) 为 $16 \times 16$，那么一张图就被切成了 $14 \times 14 = 196$ 个 Patch，现在的输入序列长度 $N = 196$

- 拉平与投影：

  切块后每个 Patch 还是一个 3D 的像素块（$16 \times 16 \times 3$ 个通道），接下来要将像素块变为固定维度的向量

  - 拉平：把 $16 \times 16 \times 3$ 的像素块展平成一个长度为 $768$ 的长向量 

  - 线性投影 (Linear Projection)：用一个全连接层（矩阵乘法），把这个长向量映射到 Transformer 需要的维度 $D$

    相当于把“像素特征”转换成了“Embedding 特征”，论文中称为Patch Embeddings

- 加入特殊身份 [Class] Token

​			在196个向量序列前加入一个可学习向量$x_{class}$，序列长度由196变化为197

​			目的：作者人为规定这个加入的token的输出才代表整张图片的分类结果，其余的输出在分类任务中被丢弃

- 注入位置信息-----position embedding

  transformer的注意力机制是全局的，所以如果顺序打乱，算出来的结果是一样的。因此每个embedding要加入位置向量

​				

### 归纳偏置（导致ViT在小数据集上表现不如resnet，却在大数据集表现很好）

##### CNN：生来就知道局部性和平移等变性，所以是强归纳偏置

##### ViT：对图像一无所知，左上角patch和右下角patch距离感相同，所以需要学习空间关系，只有在切分patch和加入位置编码才有唯一的归纳偏置

### ViT实操指南

- 预训练pre-training：用小一些的分辨率
- 微调：用大一点的分辨率，为了看的更清楚，效果更好

遇到的问题：

- patch大小不变：通常固定patch大小为16*16
- 序列边长：$224 \times 224$ 的图 $\rightarrow$ $14 \times 14 = 196$ 个 Patch， $384 \times 384$ 的图 $\rightarrow$ $24 \times 24 = 576$ 个 Patch
- 在预训练中训练的196个位置编码，现在有576个patch。如何对应
- 解决办法：
  - 训练好的 196 个位置向量，不再看作一长串 (1D)，而是按照原本的图片位置排好，恢复成 $14 \times 14$ 的网格
  - 据新的分辨率（比如 $24 \times 24$），把这个 $14 \times 14$ 的位置编码网格进行**双线性插值 **，硬生生地把它“拉大”去适应新的尺寸 

### ViT代码

```python
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    对应论文 Section 3.1: 将 2D 图像切块并投影为 1D 向量序列
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        # 计算 Patch 的总数量 N = (H/P) * (W/P)
        # 例如 224/16 = 14, 14*14 = 196
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

        # 关键点：用 Conv2d 来实现切块+投影
        # kernel_size=16, stride=16 意味着每 16x16 的像素块被卷积成一个 1x1 的点（但有 embed_dim 个通道）
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x 的形状: [Batch, 3, 224, 224]
        x = self.proj(x)  
        # 经过卷积后形状: [Batch, 768, 14, 14]
        
        # 展平 H 和 W 维度: [Batch, 768, 196]
        x = x.flatten(2)
        
        # 交换维度以适应 Transformer 输入: [Batch, 196, 768]
        x = x.transpose(1, 2)
        return x
```



```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, 
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        
        # 1. Patch Embedding 层
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        # 2. Class Token (对应论文 Eq. 1 中的 x_class)
        # 形状为 [1, 1, 768]，是一个可学习的参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 3. Position Embedding (对应论文 Eq. 1 中的 E_pos)
        # 形状为 [1, 196+1, 768]，也是可学习的参数
        # 注意：这里是 196 + 1，因为多了一个 class token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # 4. Dropout (论文中提到在 embedding 后加 dropout)
        self.pos_drop = nn.Dropout(p=0.1)

        # 5. Transformer Encoder
        # PyTorch 自带的标准 Transformer 层 (对应论文 Eq. 2 & 3)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            activation="gelu",  # 论文特别提到用 GELU
            batch_first=True,
            norm_first=True     # 现代 ViT 实现通常把 Norm 放在 Attention 前 (Pre-Norm)
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 6. Layer Norm (在 MLP Head 之前还有一层 Norm，对应 Eq. 4)
        self.norm = nn.LayerNorm(embed_dim)

        # 7. MLP Head (分类头)
        self.head = nn.Linear(embed_dim, num_classes)

        # 初始化权重 (省略细节，通常是从截断正态分布初始化)
        self._init_weights()

    def _init_weights(self):
        # 简化的初始化
        nn.init.normal_(self.pos_embed, std=.02)
        nn.init.normal_(self.cls_token, std=.02)
        nn.init.xavier_uniform_(self.head.weight)

    def forward(self, x):
        # --- Step 1: 图像变序列 ---
        # x: [Batch, 3, 224, 224] -> [Batch, 196, 768]
        x = self.patch_embed(x)

        # --- Step 2: 加上 Class Token ---
        # 复制 cls_token 到当前 batch 大小: [Batch, 1, 768]
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 拼接: [Batch, 197, 768]
        x = torch.cat((cls_token, x), dim=1)

        # --- Step 3: 加上 Position Embedding ---
        # 广播机制自动相加: [Batch, 197, 768]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # --- Step 4: Transformer 编码 ---
        # 经过 12 层处理，形状不变: [Batch, 197, 768]
        x = self.blocks(x)

        # --- Step 5: 分类 ---
        # 对应 Eq. 4: y = LN(z_L^0)
        # 这一步体现了 Norm 和 只取第 0 个 token (cls_token) 的输出
        x = self.norm(x)
        cls_token_final = x[:, 0] # [Batch, 768]
        
        # 最后的线性分类
        x = self.head(cls_token_final)
        return x
```


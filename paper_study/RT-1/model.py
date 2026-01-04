import torch
import torch.nn as nn

# FiLM 层：早期语言融合
class FiLMLayer(nn.Module):
    def __init__(self, num_features, text_emb_dim=512):
        super().__init__()
        # 两个全连接层分别生成 gamma (缩放) 和 beta (平移)
        # 论文中提到：Initialize weights of dense layers to zero 
        self.fc_gamma = nn.Linear(text_emb_dim, num_features)
        self.fc_beta = nn.Linear(text_emb_dim, num_features)

        self._init_weights()

    def _init_weights(self):
        # 恒等初始化：让初始状态下 feature_new = feature_old
        # Gamma 的权重和偏置都设为 0 -> 输出为 0 -> 公式里用 (1 + gamma)
        nn.init.zeros_(self.fc_gamma.weight)
        nn.init.zeros_(self.fc_gamma.bias)
        
        # Beta 的权重和偏置都设为 0 -> 输出为 0
        nn.init.zeros_(self.fc_beta.weight)
        nn.init.zeros_(self.fc_beta.bias)

    def forward(self, feature_map, text_emb):
        # feature_map: [Batch, Channel, Height, Width]
        # text_emb: [Batch, Text_Dim]
        
        # 1. 生成调制参数
        gamma = self.fc_gamma(text_emb)  # [B, C]
        beta = self.fc_beta(text_emb)    # [B, C]

        # 2. 调整维度以匹配特征图 (广播机制)
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        beta = beta.view(beta.size(0), beta.size(1), 1, 1)

        # 3. 应用仿射变换 (Feature-wise Linear Modulation)
        # 公式: F_new = (1 + gamma) * F_old + beta
        return (1 + gamma) * feature_map + beta

# 伪代码：集成到 EfficientNet Block 中
class FiLM_MBConvBlock(nn.Module):
    def forward(self, x, text_emb):
        x = self.conv1(x)
        x = self.bn1(x)
        # 在卷积和激活之间或 Block 之后插入 FiLM
        x = self.film(x, text_emb) 
        x = self.act(x)
        # ... 后续操作
        return x
    


# TokenLearner：空间注意力压缩
class TokenLearner(nn.Module):
    def __init__(self, in_channels, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        
        # 一个轻量级的卷积网络，用于生成 Attention Maps
        # 输入通道是 C，输出通道是 num_tokens (即生成的 Token 数量)
        self.attention_generator = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels), # Depthwise
            nn.GELU(),
            nn.Conv2d(in_channels, num_tokens, kernel_size=1), # Pointwise，输出 8 个通道
            nn.Sigmoid() # 生成 0-1 之间的注意力权重，也可以用 Spatial Softmax
        )

    def forward(self, feature_map):
        # feature_map: [Batch, Channel=512, H=9, W=9] [cite: 136]
        
        # 1. 生成 8 个空间注意力图
        # shape: [Batch, 8, 9, 9]
        attn_maps = self.attention_generator(feature_map)
        
        # 2. 归一化 (让每个 map 在空间维度 H*W 上和为 1，类似 Softmax)
        # 这一步在某些实现中可选，但有助于稳定梯度
        B, C, H, W = feature_map.shape
        attn_maps = attn_maps.view(B, self.num_tokens, H * W)
        attn_maps = torch.softmax(attn_maps, dim=-1) 
        attn_maps = attn_maps.view(B, self.num_tokens, H, W)

        # 3. 广播并点乘聚合 (Element-wise Multiplication & Sum)
        # 我们需要计算每个 token 对应的特征向量
        # feature_map: [B, C, H, W] -> [B, 1, C, H, W]
        # attn_maps:   [B, N, H, W] -> [B, N, 1, H, W]
        
        feat = feature_map.unsqueeze(1) 
        attn = attn_maps.unsqueeze(2)
        
        # 结果: [B, N, C, H, W]
        weighted_feat = feat * attn
        
        # 4. 在空间维度 (H, W) 上求和，得到最终 Token
        # tokens: [Batch, Num_Tokens=8, Channel=512]
        tokens = weighted_feat.sum(dim=[-2, -1])
        
        return tokens
    
# 离散化分类输出
# RT-1 不回归连续值，而是做分类。动作空间有 11 个维度，每个维度 256 个类别 。
class ActionHead(nn.Module):
    def __init__(self, transformer_dim=512, num_bins=256):
        super().__init__()
        # 11 个维度：7 (arm) + 3 (base) + 1 (mode)
        self.action_dims = 11 
        self.num_bins = num_bins
        
        # 一个简单的线性层，将 Transformer 输出映射到所有动作维度的 logits
        # 输出大小: 11 * 256
        self.proj = nn.Linear(transformer_dim, self.action_dims * num_bins)

    def forward(self, transformer_output):
        # transformer_output: [Batch, Seq_Len, Dim]
        # 我们只取最后一个时间步的输出作为当前动作预测
        last_token = transformer_output[:, -1, :] 
        
        # 生成 logits
        logits = self.proj(last_token) # [Batch, 11 * 256]
        
        # 重塑形状以便计算 Loss
        # [Batch, 11, 256]
        logits = logits.view(-1, self.action_dims, self.num_bins)
        
        return logits

# 训练时的 Loss 计算
def compute_loss(pred_logits, target_actions):
    # pred_logits: [B, 11, 256]
    # target_actions: [B, 11] (已经是 0-255 的离散整数索引)
    
    loss_fn = nn.CrossEntropyLoss()
    
    # 需要对每个动作维度分别计算 Loss 然后求和
    total_loss = 0
    for i in range(11):
        total_loss += loss_fn(pred_logits[:, i, :], target_actions[:, i])
        
    return total_loss


class RT1(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 文本编码 (预训练好的 USE，通常冻结或微调)
        self.text_encoder = UniversalSentenceEncoder() 
        
        # 2. 视觉主干 (ImageNet 预训练 + FiLM)
        self.backbone = EfficientNetB3_With_FiLM() 
        
        # 3. 压缩器
        self.token_learner = TokenLearner(in_channels=512, num_tokens=8)
        
        # 4. 序列模型 (Decoder-only Transformer)
        # 输入长度 = 6 images * 8 tokens = 48 tokens
        self.transformer = TransformerDecoder(layers=8, dim=512) 
        
        # 5. 输出头
        self.action_head = ActionHead()

    def forward(self, video_frames, text_instruction):
        # video_frames: [Batch, Time=6, C, H=300, W=300]
        # text_instruction: list of strings
        
        B, T, C, H, W = video_frames.shape
        
        # --- A. 文本处理 ---
        text_emb = self.text_encoder(text_instruction) # [B, 512]
        
        # --- B. 视觉特征提取 (融合文本) ---
        # 既然 EfficientNet 处理单帧，我们将 Time 维合并到 Batch 维
        # [B*T, C, H, W]
        flat_frames = video_frames.view(B * T, C, H, W)
        
        # 文本嵌入也需要广播到每一帧
        flat_text_emb = text_emb.repeat_interleave(T, dim=0) # [B*T, 512]
        
        # 输出特征图: [B*T, 512, 9, 9]
        # 在这里，FiLM 层已经在 backbone 内部发挥作用了
        feature_maps = self.backbone(flat_frames, flat_text_emb)
        
        # --- C. Token 压缩 ---
        # 输出: [B*T, 8, 512]
        visual_tokens = self.token_learner(feature_maps)
        
        # --- D. 序列建模 ---
        # 恢复 Batch 和 Time 维度，形成 Transformer 的输入序列
        # [B, T*8, 512] -> [B, 48, 512]
        transformer_input = visual_tokens.view(B, T * 8, 512)
        
        # 加上 Positional Embedding (代码省略)
        
        # Transformer 处理
        # [B, 48, 512]
        transformer_out = self.transformer(transformer_input)
        
        # --- E. 动作预测 ---
        # [B, 11, 256]
        action_logits = self.action_head(transformer_out)
        
        return action_logits
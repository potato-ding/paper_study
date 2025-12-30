import torch
import torch.nn as nn
import torch.nn.functional as F
# 文本流、视觉流结构相同，但参数不共享
class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_layers=2, num_heads=4):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers)

    def forward(self, x, mask=None):
        return self.encoder(x, src_key_padding_mask=mask)

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, query, key_value, key_padding_mask=None):
        """
        query:       (B, Tq, D)
        key_value:   (B, Tk, D)
        """
        attn_out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask
        )
        # 对应论文Q = H_text  K,V = H_vision
        # residual + norm
        return self.norm(query + attn_out)
    
#  跨模态交叉注意力， ViLBERT最大创新点
class CoAttentionLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.text_to_vision = CrossAttention(dim)
        self.vision_to_text = CrossAttention(dim)

    def forward(self, text_feat, vision_feat, text_mask=None, vision_mask=None):
        """
        text_feat:   (B, T_text, D)
        vision_feat:(B, T_vis, D)
        """
        # Text attends to Vision
        text_out = self.text_to_vision(
            query=text_feat,
            key_value=vision_feat,
            key_padding_mask=vision_mask
        )

        # Vision attends to Text
        vision_out = self.vision_to_text(
            query=vision_feat,
            key_value=text_feat,
            key_padding_mask=text_mask
        )

        return text_out, vision_out
    
# 拼接成ViLBERT
class MiniViLBERT(nn.Module):
    def __init__(self, vocab_size=30522, dim=256, max_text_len=16, max_vis_len=8):
        super().__init__()

        # --- Embeddings ---
        self.word_emb = nn.Embedding(vocab_size, dim)
        self.text_pos_emb = nn.Embedding(max_text_len, dim)

        self.vis_emb = nn.Linear(2048, dim)   # fake region feature
        self.vis_pos_emb = nn.Embedding(max_vis_len, dim)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.img_token = nn.Parameter(torch.randn(1, 1, dim))

        # --- Encoders ---
        self.text_encoder = TransformerEncoder(dim)
        self.vision_encoder = TransformerEncoder(dim)

        self.co_attn = CoAttentionLayer(dim)

        # --- Task head ---
        self.classifier = nn.Linear(dim, 1)

    def forward(self,
                input_ids,        # (B, T_text)
                text_mask,        # (B, T_text)
                vision_feats,     # (B, T_vis, 2048)
                vision_mask):     # (B, T_vis)

        B, T_text = input_ids.shape
        B, T_vis, _ = vision_feats.shape

        # ---- Text embedding ----
        pos_ids = torch.arange(T_text, device=input_ids.device)
        pos_ids = pos_ids.unsqueeze(0).expand(B, T_text)

        text_emb = (
            self.word_emb(input_ids) +
            self.text_pos_emb(pos_ids)
        )

        cls = self.cls_token.expand(B, -1, -1)
        text_emb = torch.cat([cls, text_emb], dim=1)
        text_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool), text_mask],
            dim=1
        )

        # ---- Vision embedding ----
        vision_emb = self.vis_emb(vision_feats)

        img = self.img_token.expand(B, -1, -1)
        vision_emb = torch.cat([img, vision_emb], dim=1)
        vision_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool), vision_mask],
            dim=1
        )

        # ---- Intra-modality encoding ----
        text_feat = self.text_encoder(text_emb, text_mask)
        vision_feat = self.vision_encoder(vision_emb, vision_mask)

        # ---- Cross-modality fusion ----
        text_feat, vision_feat = self.co_attn(
            text_feat, vision_feat,
            text_mask, vision_mask
        )
        # ---- Late fusion (CLS ⊙ IMG) ----
        text_cls = text_feat[:, 0]
        vision_cls = vision_feat[:, 0]

        joint = text_cls * vision_cls
        return self.classifier(joint).squeeze(-1)


def fake_batch(batch_size=4,
               text_len=10,
               vis_len=5,
               vocab_size=30522):

    input_ids = torch.randint(0, vocab_size, (batch_size, text_len))
    text_mask = torch.zeros(batch_size, text_len, dtype=torch.bool)

    vision_feats = torch.randn(batch_size, vis_len, 2048)
    vision_mask = torch.zeros(batch_size, vis_len, dtype=torch.bool)

    labels = torch.randint(0, 2, (batch_size,)).float()

    return input_ids, text_mask, vision_feats, vision_mask, labels

model = MiniViLBERT()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for step in range(20):
    batch = fake_batch()
    input_ids, text_mask, vision_feats, vision_mask, labels = batch

    logits = model(
        input_ids,
        text_mask,
        vision_feats,
        vision_mask
    )

    loss = F.binary_cross_entropy_with_logits(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"step {step:02d} | loss = {loss.item():.4f}")
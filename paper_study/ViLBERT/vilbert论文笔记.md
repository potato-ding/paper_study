# vilbert论文笔记

### 总结

Two-Stream + Co-Attention + Pretrain Visual Grounding

用预训练的方式学习“视觉-语言对齐（visual grounding）”，并证明这种能力可以迁移到多种下游任务

### 解决的问题

每个任务都设计一个**专用模型**，视觉与语言的“对齐”**只在下游任务中学**，所以泛化差、依赖标注、不可迁移

Visual grounding （视觉语言对齐）本身应该是可以“预训练”的通用能力，而不是任务附属品。

### 核心设计

-  Two-Stream：显示对齐（预训练中）

  分为text stream流和visual stream流分别self-attention，在进行co-attention后文本 token 已经带上“相关图像信息”，图像 region已经带上“相关语言信息”，在下游任务中才会真正合成一个表示

  ```
  Text Stream (Language)              Visual Stream (Vision)
  ─────────────────────              ──────────────────────
  
  w1  w2  w3 ...                      v1  v2  v3 ...
   │   │   │                          │   │   │
   ▼   ▼   ▼                          ▼   ▼   ▼
  ┌──────────────┐                  ┌──────────────┐
  │  TRM Layer   │                  │  TRM Layer   │
  │ (Self-Attn)  │                  │ (Self-Attn)  │
  └──────────────┘                  └──────────────┘
          │                                  │
          ▼                                  ▼
  ┌────────────────────┐    Co-Attention   ┌────────────────────┐
  │   Co-TRM (Text)    │◀────────────────▶│  Co-TRM (Vision)   │
  │ Q: Text            │                  │ Q: Vision          │
  │ K,V: Vision        │                  │ K,V: Text          │
  └────────────────────┘                  └────────────────────┘
          │                                  │
          ▼                                  ▼
  ┌──────────────┐                  ┌──────────────┐
  │  TRM Layer   │                  │  TRM Layer   │
  └──────────────┘                  └──────────────┘
          │                                  │
         ...                                ...
  
  
  single stream
  Image Regions        Text Tokens
   v1  v2  v3 ...      w1  w2  w3 ...
     │   │   │           │   │
     └───┴───┴──────┬────┴───┴──────┐
                    ▼
          [ Concatenate Tokens ]
          (IMG, v1, v2, ..., CLS, w1, w2, ...)
                    │
          ┌───────────────────────┐
          │   Transformer Layer   │
          │   (Self-Attention)    │
          └───────────────────────┘
                    │
          ┌───────────────────────┐
          │   Transformer Layer   │
          │   (Self-Attention)    │
          └───────────────────────┘
                    │
                   ...
                    │
          ┌───────────────────────┐
          │   Transformer Layer   │
          └───────────────────────┘
  ```

- 对其判别：

  输入：image + caption

  输出：score = h_IMG ⊙ h_CLS

  直接教模型做跨模态对齐

### 结论

- Two-Stream > Single-Stream
- ViL BERT的预训练很重要
- 可迁移
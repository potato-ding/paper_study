# RT-1精读

### 核心概览

解决的问题：如何将大模型、多任务的transformer模型应用于现实世界的机器人控制，并实现高泛化能力

- 核心贡献：提出了模型架构RT-1，能够以3Hz的频率在移动机械臂上实行运行
- 数据规模：13台机器人17个月手机的130k条演示数据，包含700多项任务
- 结论：该模型不仅能吸收大规模数据，还能通过混合仿真数据和其他机器人的数据来获得新的技能

### 模型架构

#### 输入与输出

- 输入：
  - 图像：历史 6 帧图像（$300 \times 300$ 分辨率）
  - 文本：自然语言指令
- 输出：离散化的动作 Token（Action Tokens）。动作空间包含 7 个维度的机械臂运动（x, y, z, roll, pitch, yaw, gripper）、3 个维度的底盘运动以及 1 个模式切换变量 。

注意：RT-1 将每个动作维度离散化为 256 个 bin，这与通常使用连续高斯分布的机器人策略不同 。消融实验证明，这种离散化对于捕捉复杂的多模态动作分布至关重要 。

#### 核心组件

![image-20260104085953418](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20260104085953418.png)

##### 第一阶段：特征提取与融合

- 图像编码器：没有直接用ViT，而是用了 ImageNet 上预训练的 **EfficientNet-B3**。输入是6帧历史图像（300*300）。因为EfficientNet 推理速度快，且卷积网络对底层纹理特征提取很稳健。
- 语言指令：使用 Universal Sentence Encoder (USE) 将文本（如 "pick apple"）嵌入为向量 。
- 融合机制（FiLM）：通过FiLM层（Feature-wise Linear Modulation）来融合图文
  - 原理：语言嵌入向量被用来生成仿射变换参数（$\gamma$ 和 $\beta$），直接对 EfficientNet 中间层的特征图进行**缩放和平移**。
  - **技巧**：作者使用了 **Identity-initialized FiLM** 。初始化时，FiLM 层不起作用（保持恒等映射），这样可以保留 EfficientNet 预训练好的 ImageNet 特征，随着训练进行再慢慢注入语言信息。

##### 第二阶段：token压缩——效率的关键

问题：EfficientNet 输出的特征图是 $9 \times 9 = 81$ 个 Token。如果输入 6 帧历史图像，就是 $6 \times 81 = 486$ 个 Token。对于一个需要高频运行的机器人来说，Transformer 处理这么多 Token 太慢了

解决方案：

- 引入TokenLearner模块
- 学习一组空间注意力掩码，从每张图的81个token中，加权聚合出8个最关键的token
- 结果：transformer的输入长度从486降到6*8=48个token，使推理速度提升了2.4倍以上

##### 第三阶段：序列建模与输出

主干：标准的decoder-only transformer，包含8个self-attention，参数量为19M

输入：48个视觉token + postional encoding

输出：输出的是机械臂动作

- 动作空间：7维机械臂参数（x, y, z, roll, pitch, yaw, gripper）+ 3 维底盘参数 + 1 个模式切换   共11个维度

- 离散化：将每个维度切分成 **256 个离散的区间 (bins)** ，没有使用连续的数值。

  原因：消融实验中离散化动作表现远好于连续的高斯分布。离散化允许模型表达多模态分布（例如：面对障碍物，可以从左绕也可以从右绕，而不是取平均值撞上去）。



#### 数据海绵

模型架构不仅要好，还要能“吸收”各种各样的数据。

##### 异构数据吸收

- 通常机器人学习很难利用不同机器人的数据（因为身体结构不一样）
- RT-1 做了一个实验：把 **Kuka 机械臂**（工业手臂，只有夹爪，不能移动）的数据和 **Everyday Robots**（移动服务机器人）的数据混合训练 。
- **结果**：RT-1 居然在 Everyday Robots 上学会了 Kuka 擅长的“Bin-picking”（从杂乱篮子里抓物体）技能，且成功率翻倍 。这意味着 Transformer 成功提取了跨机器人形态的通用物理交互逻辑。



#### 关键消融实验结论

- ImageNet预训练不可或缺：如果去掉 EfficientNet 的 ImageNet 预训练，泛化到新任务的能力下降 33% 。**这说明通用视觉特征是机器人理解世界的基石。**
- **Transformer 的作用**：如果去掉 Transformer，只用 EfficientNet+MLP，面对干扰物（Distractors）和未见任务时性能会下降。Transformer 的 Self-Attention 帮助模型在时间序列中“聚焦”于关键物体，忽略背景噪音 。
- 不需要自回归动作：在文本生成中，我们是一个字一个字生成的。但在 RT-1 中，一次性并行输出 11 个动作维度的 Token 效果就很好，改成自回归反而会拖慢推理速度 2 倍，且没有性能提升 。

一、FiLM-EfficientNet

- 图像首先通过ImageNet上预训练的**EfficientNet-B3**
- 语言融合机制：使用FiLM层。语言指令先通过Universal Sentence Encoder（USE）嵌入，通过FiLM层调节EfficientNet 的特征图 
- 技术细节：RT-1 采用**早期融合**。这意味着图像特征在提取过程中就已经被语言指令“关注”了，这有助于提取与任务相关的视觉特征 。


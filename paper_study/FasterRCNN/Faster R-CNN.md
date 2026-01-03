# Faster R-CNN

### 前沿知识：Fast R-CNN

![image-20251227094709581](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20251227094709581.png)

- deep convNet的得到feature map
- 将**候选区域**通过ROI投影映射到feature map
- 通过Rol pooling成指定大小的矩阵块
- 用这个指定大小的矩阵块及性能softmax分类以及bbox位置的确认
- fast RCNN通过这个方法能达到实时的效果

![image-20251227095159575](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20251227095159575.png)

region proposal：候选区域的生成过程    这个部分非常耗时，所以Fast R-CNN并没有计算这部分的时间

![image-20251227100136860](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20251227100136860.png)

### 动机：

proposal候选框的生成过于慢，导致并不能实时得到效果

![image-20251227100158993](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20251227100158993.png)

Image → 共享卷积层 → RPN 产生 proposal → 对 proposal 做 RoI 特征抽取 → Fast R-CNN 对 proposal 进行「分类 + 边框回归」

### 贡献：

- region proposal的生成和feature map的生成都需要通过CNN特征提取，因此使用了sharing computation

- RPN：是将图片得到proposal的整个过程

  特点是：用一个全卷积网络（RPN）在共享特征图上“顺手”生成候选框，proposal 几乎零成本。

- 将proposal生成从每张图片2s加速到10ms



### 得到proposal的过程

- 得到feature map

- 使用小的网络以滑块的方式逐步得到候选区域

- 小网络内部：每个window映射到低维特征，再分别传给分类层和回归层得到类别和坐标

  ![image-20251227101831939](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20251227101831939.png)

​	其中2表示的是二分类判断类别，4表示的是坐标信息， k表示的是anchor box

- k anchor box尺寸和比例的确定

  作者使用了3个尺寸128,256,512和3个比例一共9种anchor

- loss设计：

  - 真值：最高IOU的anchor和IOU>0.7的anchor,因此可能会有多个真值
  - 负样本：IOU<0.3的anchor
  - 损失：分类损失 + 回归损失（只包含正样本）

  ![image-20251227105725606](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20251227105725606.png)

### 训练方法：

- 交替4步训练法（论文中方法）：

  - 单独训练RPN网络
  - 用RPN得到的proposal训练Fasr R-CNN
  - Fast R-CNN得到的convnet替换第一步的RPN中的convnet，保持权重不变，只训练后面模块的权重
  - 再用新的proposal训练fast R-CNN
  - RPN和Fast R-CNN的convnet相同，合并后得到Faster R-CNN

  ![image-20251227150807213](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20251227150807213.png)

- 联合训练法：

  - 近似联合训练：直接使用Faster R-CNN网络进行训练，proposal损失和fast R-CNN的损失反向传播

![image-20251227151038677](C:\Users\dy\AppData\Roaming\Typora\typora-user-images\image-20251227151038677.png)

​		作者在论文中使用上一个方法，但是后续发布的代码使用的是联合训练法，得到的结果几乎是相同的，但时间大幅度减小

缺点：Fast R-CNN使用了proposal，但是单纯的Fast RCNN的损失没有proposal的反向传播过程


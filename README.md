# GGMF
The implementation of GGMF
# Hybrid Graph Convolutional Network With Online Masked Autoencoder for Robust Multimodal Cancer Survival Prediction

2023年 TMI  医学影像分析领域的顶刊

![image-20250429160925174](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429160925174.png)

## 图注意力网络:

在为每位患者生成对应的多模态特征基础上,构建患者之间的相似图。然后通过对"图注意力的一些操作"来提升预测效果

![image-20250307142424304](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250307142424304.png)



![image-20250307152729052](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250307152729052.png)

![image-20250307152747298](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250307152747298.png)



![image-20250403181433471](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250403181433471.png)



采用独立图的方式实现相邻结点之间的消息传递。

一个超图卷积网络，利用"超边"来实现不同模态之间的信息交互。

并通过自动编码器去 生成 缺失的超边。

也就是说学的是 模态内和模态间的特征 以及 **模型在训练过程中处理意外确实模态之间的特征**

**病理图的预处理过程：**

图片切割成 512 * 512的patch，然后用KimiaNet(一个预训练的网络)来编码映射成这些patch成1024的维度。

然后根据每个原始图像块的位置信息通过邻接矩阵"绑定在一起" V是当前这个patch A是对应的八个方向的邻接关系

![image-20250429102948684](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429102948684.png)

**临床信息的预处理过程:**

V是临床记录，包括(年龄，性别，BMI，是否接受过治疗)等信息。通过one-hot编码来量化这些信息，并编码成1024维度。作者对A邻接矩阵的处理很 粗暴，直接是一个全连接，也就是所有结点之间互相连接。

![image-20250429104711548](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429104711548.png)

基因数据的预处理:

作者通过前人的方法 **GSEA**(Gene Set Enrichment Analysis) (就是人为的划分各个基因的作用) 把各个基因划分为5大类生物学功能:

![image-20250429105349088](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429105349088.png)

最终每位病人的基因数据被转化为5个结点，每个结点代表一类基因模块的表达特征。(因为深度学习的黑盒特性，其实这里很扯)V是这5个结点的特征集合。A也是很粗暴的全连接，表示每个基因功能模块都有连接，

其中由于这个过程采用GCN的方法来训练，所以可以忽略掉**部分缺失数据**

## 传统GCN

**GCN**是处理**图结构数据**的一种深度学习模型。

核心思想：**每一层**中一个节点的特征不仅来自于自己，还会结合它的**邻居节点的特征**。通过反复叠加图卷积层，节点可以获得更大范围的邻居信息，形成综合表示。

![image-20250429111116119](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429111116119.png)

就是**邻接矩阵信息加权求和+线性变换+非线性激活**

所以在更新过程中，如果某位病人的临床信息缺失了，GCN还是能继续更新

## 处理过程中对模态内之间的处理:

![image-20250429112245916](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429112245916.png)

每个模态都会单独走一个完全一样的子网络:

这个子网络是由三层 GraphSAGE 堆叠起来的。这个 GraphSAGE 是其他作者提的结构。其实就是用这个以图结构为基础的网络来 提特征。

### 后续单个模态中信息交互的方法:

![image-20250429134918246](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429134918246.png)

![image-20250429134931531](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429134931531.png)

![image-20250429135004369](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429135004369.png)可以理解为 是单个模态融合后的信息。其实细看还是常规的特征加权。

## 处理过程中对不同模态之间的处理:

![image-20250429140102103](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429140102103.png)

上述过程中![image-20250429141246824](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429141246824.png)是所有![image-20250429141307710](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429141307710.png)的集合。这个处理特征其实是很常规的方法。

然后最终的特征是一个 信息交互前的特征 元素级相加 信息交互后的特征。

![image-20250429142537107](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429142537107.png)

## **子任务**

其中有一个自动编码器来 补全单个模态的特征信息。可以理解为借助 数据的特征分布来 拟合 缺失信息。

![image-20250429151522298](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429151522298.png)

值得做注意的是，他不像传统的生成式对抗网络直接生成原始数据，比如图片，而是直接生成符合这个数据分布的特征

具体的处理过程类似 "生成式对抗网络"

![image-20250429152019738](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429152019738.png)

随机丢弃模态的一部分特征，视为缺失，模型无法直接访问他们。然后将剩余特征输入到模型进行编码。可以理解为这个过程学到的是一种隐含的“映射关系”。

然后，模型尝试重新恢复被丢弃的数据，这个时候我们将原本丢失的数据特征和模型生成的特征 做"差"，我们在最小化这个"差值"的过程中更新参数，最终让模型学到了 这个映射过程。

![image-20250429152538502](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429152538502.png)

上式是这个子任务所使用的损失函数  B代表Batch Size M表示模态的数量 C表示超边的数量![image-20250429153410310](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429153410310.png)代表真实的特征，后面那个是模型生成的特征。

![image-20250429153620013](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429153620013.png)

似然估计是"生存分析"共用的损失函数

![image-20250429153733359](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429153733359.png)

这是总的损失函数，通过两个超参来平衡 各个任务在更新过过程中的 比例

![image-20250429153807505](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429153807505.png)

## 实验

对比:

![image-20250429154827807](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429154827807.png)

常规的与单模态 和 多模态 之间的对比。

消融:

![image-20250429155008687](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429155008687.png)

两个消融，也就是将GCN替换成线性层以及去除模态之间的交互。

![image-20250429155152790](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429155152790.png)

可解释性 一致性的做法 可视化病理图的注意力权重

此外为了验证 模型子任务的 生成特征的 结果，采用了三种 特征重建 的对比方法

![image-20250429155744619](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429155744619.png)

![image-20250429155837466](C:\Users\hello_xmj\AppData\Roaming\Typora\typora-user-images\image-20250429155837466.png)

总得来说在用图神经网络编码特征 之后 就是常规的 特征处理方法。有能 改动的 地方。比如上面那个 GraphSAGE，以及子任务的特征重建。其实文中也明确说了 这两个模块是借鉴别人的工作。

# Convolutional Neural Networks
## 引言

1. **回顾MLP（全连接层）的局限性**
(I) 传统方法是将图像**展平（flattening）** 为一维向量来处理。
(II) 这种做法的致命缺陷是**忽略了图像的空间结构**，即像素之间的空间关系。
(III) MLP对特征的顺序（像素的排列）很敏感，但我们希望模型能利用“邻近像素更相关”这一事实。

2. **卷积神经网络（CNN）的引入**
(I) CNN是为精确处理这类具有空间结构的数据而设计的专用神经网络。
(II) 它们的设计灵感来源于生物学、群论和大量的实验探索。
(III) CNN现在是计算机视觉领域的基石，并在ImageNet竞赛后（Krizhevsky et al., 2012）变得无处不在。

3. **CNN的核心优势**
(I) **计算效率高**：相比全连接网络，CNN通常需要更少的参数。
(II) **样本效率高**（Sample Efficient）：由于参数共享，它们对训练数据的需求更少。
(III) 易于在GPU上**并行计算**。
(IV) CNN不仅限于2D图像，也已成功应用于1D序列（如音频、文本、时间序列）和图结构数据。

4. **本章（第7章）学习路线图**
(I) 首先，深入探讨CNN的**动机**。
(II) 接着，学习构成所有CNN骨干的**基本操作**：包括卷积层、填充（Padding）和步幅（Stride）、用于聚合信息的池化层（Pooling Layers）以及多通道（Multiple Channels）的使用。
(III) 讨论现代CNN架构的结构。
(IV) 最后，以一个完整的**LeNet**（第一个成功部署的CNN）实例结束本章。

5. **展望**
(I) 下一章（第8章）将深入探讨更流行、更近期的CNN架构。

---

## 7.1 From Fully Connected Layers to Convolutions

1.  **回顾：全连接层（MLP）的适用性**
(I) 全连接层（MLP）至今仍是处理**表格数据（tabular data）** 的合适选项。
(II) 在表格数据中，我们可能预期特征之间存在交互，但MLP**不会假设任何关于特征如何交互的先验结构**。

2.  **全连接层（MLP）处理图像的挑战：参数爆炸**
(I) 对于高维感知数据（如图像），MLP的结构化缺陷会变得非常棘手。
(II) **一个例子**：处理 $1000 \times 1000$ （一百万像素）的图像。
(III) 如果输入层有一百万个维度，隐藏层有1000个单元，那么仅这一个隐藏层就会有 $10^6 \times 10^3 = 10^9$（十亿）个参数。
(IV) 即使拥有大量GPU和耐心，学习这么多参数在现实中也是**不可行的（infeasible）**。

3.  **问题的关键：利用结构**
(I) 尽管参数量巨大，但人类和计算机都能很好地区分猫和狗，这表明图像中存在可利用的丰富结构。
(II) **卷积神经网络（CNNs）** 就是一种被创造出来专门**利用自然图像中已知结构** 的机器学习模型。

### 7.1.1 Invariance(不变性)

1.  **核心思想：空间不变性（Spatial Invariance）**
(I) 我们用来识别物体的方法，理论上**不应该过度依赖物体在图像中的“精确位置”**。
(II) **以“Where's Waldo”（威利在哪里）为例**：
    --- (A) 找到 Waldo 之所以困难，是因为存在大量干扰物。
    --- (B) 但一个关键事实是：**Waldo "长什么样"（What Waldo looks like）与他 "在哪里"（where Waldo is located）无关**。
    --- (C) 我们可以想象用一个“Waldo 检测器”扫描（sweep）图像的每个小块（patch）并打分。
(III) CNNs 系统化了**空间不变性**这一思想，使其能用更少的参数学习到有用的表示。

2.  **构建计算机视觉模型的三个期望属性（Desiderata）**
(I) **平移不变性（Translation Invariance）**：
    --- (A) 在网络的最初几层，无论一个**相同的图像块（patch）** 出现在图像的哪个位置，网络都应该产生 **相似的响应**。
    --- (B) 这有时也被称为**平移等变性（translation equivariance）**。
(II) **局部性原则（Locality Principle）**：
    --- (A) 网络的最初几层应该**只关注局部区域**，而忽略图像中距离较远区域的内容。
    --- (B) 这些局部表示最终将在更高层被聚（aggregate），以对整个图像进行预测。
(III) **层次结构（Hierarchy）**：
    --- (A) 随着网络层数的加深，后续的层应该能够捕捉到 **更大范围（longer-range）** 的特征，这类似于生物的视觉从低级到高级的感知过程。

### 7.1.2  Constraining the MLP(约束 MLP)

1.  **起点：全连接层 (MLP) 的困境**
(I) 我们可以将2D图像 `X` 和隐藏表示 `H` 都视作矩阵。
(II) 如果使用全连接层，如公式 (7.1.1) 所示，权重 `[V]i,j,a,b` 会同时依赖于**输出位置 (i, j)** 和**偏移量 (a, b)**。
    $$[H]_{i,j} = [U]_{i,j} + \sum_{a} \sum_{b} [V]_{i,j,a,b} [X]_{i+a,j+b}$$
(III) **代价**：对于 $1000 \times 1000$ 的图像，这一层需要 $10^{12}$ (万亿级别) 个参数，这在计算上是不可行的。

2.  **第一个约束：平移不变性 (Translation Invariance)**
(I) 我们引入的第一个原则：输入 `X` 的平移应该只导致输出 `H` 的平移。
(II) **实现方式**：权重 `V` 和偏置 `U` **不能**依赖于位置 `(i, j)`。
(III) **约束结果**：我们将 `[V]i,j,a,b` 简化为 `[V]a,b`，并将 `[U]i,j` 简化为常数 `u`。
(IV) 这就得到了**卷积**（公式 7.1.2）：
    $$[H]_{i,j} = u + \sum_{a} \sum_{b} [V]_{a,b} [X]_{i+a,j+b}$$
(V) **效果**：参数量不再依赖于图像大小，急剧下降（例如从 $10^{12}$ 降至 $4 \times 10^6$）。这就是**参数共享（Parameter Sharing）**。

3.  **第二个约束：局部性 (Locality)**
(I) 我们引入的第一个原则：我们不需要看很远就能获取 $(i, j)$ 位置的相关信息。
(II) **实现方式**：假设当偏移量 $a$ 或 $b$ 大于某个值 $\Delta$ 时（例如 $\Delta = 10$），我们设置 `[V]a,b = 0`。
(III) **约束结果**：我们只在一个很小的邻域（由 $\Delta$ 定义）内进行求和（公式 7.1.3）：
    $$[H]_{i,j} = u + \sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} [V]_{a,b} [X]_{i+a,j+b}$$
(IV) **效果**：参数量进一步从 $4 \times 10^6$ 降低到 $\approx 4\Delta^2$（如果 $\Delta=10$，大约是400个参数）。

4.  **总结**
(I) 公式 (7.1.3) 就是一个 **“卷积层”（Convolutional Layer）**。
(II) `V` 被称为**卷积核（convolution kernel）** 或 **滤波器（filter）**。
(III) 我们为参数的**大幅减少**所付出的**代价**（也是我们想要的）是：
    --- (A) 我们的特征现在是**平移不变的**。
    --- (B) 我们的层在确定每个激活值时，**只能合并局部信息**。
(IV) 事实证明，这种假设（或称为“偏置”）非常符合自然图像的特性，能让模型更好地泛化到未见过的数据。


### 7.1.3 卷积 (Convolutions)

1.  **回顾数学上的严格定义**
(I) 在数学中，两个函数（如 $f, g$）的卷积，衡量的是当一个函数被 **“翻转”（flipped）** 并移位时，两者之间的重叠。
(II) 对于二维张量，数学上的**卷积**定义如公式 (7.1.6) 所示：
    $$(f * g)(i, j) = \sum_{a} \sum_{b} f(a, b)g(i - a, j - b)$$

2.  **与我们之前公式 (7.1.3) 的区别**
(I) 这个严格定义 (7.1.6) 看上去与我们之前的公式 (7.1.3) 很像，但有一个**重大区别**。
(II) 公式 (7.1.6) 使用的是 $g(i - a, j - b)$。
(III) 而公式 (7.1.3)（我们称之为“卷积层”）使用的是 $X(i + a, j + b)$。

3.  **互相关 (Cross-correlation)**
(I) 事实上，我们之前在 (7.1.3) 中定义的运算（即深度学习中常用的“卷积”），从数学上讲，更恰当的名称应该是 **“互相关” (cross-correlation)**。
(II) 尽管有这种区别，但作者指出这种区分 **“在很大程度上只是表面上的 (mostly cosmetic)”** ，因为我们总可以通过调整记号来匹配两者（例如，一个被翻转的卷积核）。

### 7.1.4 Channels

1.  **先前模型的问题**
(I) 我们之前“天真地”忽略了一个事实：图像（例如 RGB 图像）不是二维对象，而是具有**三个通道**（红、绿、蓝）的**三阶张量**（third-order tensors）。
(II) 它们的维度是（高度, 宽度, 通道(Channel)）。
(III) 因此，输入 `X` 应表示为 `[X]i,j,k`（其中 `k` 是通道索引）。

2.  **扩展隐藏层：特征图（Feature Maps）**
(I) 就像输入是三阶张量一样，将**隐藏表示** `H` 也设置为三阶张量是一个好主意。
(II) 这意味着，在每个空间位置 `(i, j)`，我们不再只有一个隐藏值，而是有**一整个向量**的隐藏表示。
(III) 这些堆叠在一起的二维网格被称为**通道**（channels）或**特征图**（feature maps）。
(IV) **直觉**：在靠近输入的低层，不同的通道可以被专门训练来识别不同的模式（例如，一些通道识别边缘，另一些识别纹理）。

3.  **适应卷积核：从 2D 到 4D**
(I) 为了处理多通道输入 `X` 和多通道输出 `H`，我们的卷积核 `V` 必须从一个二维张量（`[V]a,b`）扩展为一个**四阶张量**：`[V]a,b,c,d`。
(II) 这四个索引的含义是：
    --- (A) `a, b`：空间偏移量（核的高度和宽度）。
    --- (B) `c`：**输入通道**（input channels）的索引。
    --- (C) `d`：**输出通道**（output channels）的索引。

4.  **多通道卷积的通用定义 (公式 7.1.7)**
(I) 综合起来，我们得到了多通道卷积层的最终定义：
    $$[H]_{i,j,d} = \sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} \sum_{c} [V]_{a,b,c,d} [X]_{i+a,j+b,c}$$
(II) **公式解读**：
    --- (A) `[H]i,j,d`：我们正在计算输出图像**第 d 个通道**上 `(i, j)` 位置的值。
    --- (B) $\sum_c$：这个求和符号是关键。它意味着，要计算**一个**输出通道 `d`，我们必须在**所有**的**输入通道** `c` 上进行卷积运算，然后将它们的结果**相加**。

5.  **展望**
(I) 这个公式 (7.1.7) 是一层卷积的通用定义，但还有很多问题需要解决。
(II) 我们还需要学习：
    --- (A) 如何将所有隐藏表示组合成一个单一的输出（例如，判断“Waldo 是否在图像中的任何地方”）。
    --- (B) 如何高效地计算。
    --- (C) 如何组合多个层、使用什么**激活函数**，以及如何做出合理的设计选择。



### 7.1.5 总结与讨论 (Summary and Discussion)

1.  **CNNs的推导**
(I) 我们从**第一性原理**（first principles）出发，推导出了卷积神经网络的结构。
(II) 事实证明，当应用合理的原则来设计图像处理算法时，CNN是**正确的选择**（right choice）。

2.  **回顾两个核心原则及其影响**
(I) **平移不变性（Translation Invariance）**：
    --- (A) 保证了图像中的所有图块（patches）都将以**相同的方式**被处理。
(II) **局部性（Locality）**：
    --- (A) 意味着在计算隐藏表示时，只使用一个**小的邻域**（small neighborhood）像素。
(III) **参数缩减**：
    --- (A) 我们展示了如何在特定假设下，**大幅减少参数**数量，同时不限制模型的**表达能力**（expressive power）。
    --- (B) 这种约束将一个在计算上**不可行**（infeasible）的问题，转变为一个**易于处理**（tractable）的模型。

3.  **通道（Channels）的角色**
(I) 引入通道，允许我们**恢复**（bring back）一部分因“局部性”和“平移不变性”约束而**失去的复杂性**。
(II) 很多现实中的图像（如高光谱图像）可以有几十到几百个通道，远不止RGB三通道。

4.  **后续展望**
(I) 在本章的后续内容中，我们将学习如何：
    --- (A) 使用卷积来有效**操纵图像的维度**。
    --- (B) 如何从**基于位置**的表示转移到**基于通道**的表示。
    --- (C) 如何高效地处理**大量的类别**。

---

## 7.2 Convolutions for Images
我们将继续使用 Img作为我们的样例.

``` python
import torch
from torch import nn
from d2l import torch as d2l
```

### 7.2.1 The Cross-Correlation Operation(互相关运算)

1.  **“卷积层”的真实身份**
(I) 严格来说，“卷积层”是一个**用词不当**（misnomer）。
(II) 它们所执行的操作，更准确地应被称为**互相关**（cross-correlations）。
(III) 在本节中，我们暂时忽略通道（channels），只看二维数据。

2.  **什么是互相关运算**
(I) 互相关是一种“滑动窗口”操作。
(II) 我们使用一个**卷积核**（kernel）在**输入张量**（input tensor）上滑动，从左到右，从上到下。
(III) 在每个位置，输入子张量和卷积核进行**逐元素**（elementwise）相乘，然后**求和**。
(IV) 这个求和的结果，就是输出张量在该位置的单个标量值。

3.  **计算示例（公式 7.2.1）**
(I) **输入 (3x3) * 卷积核 (2x2) = 输出 (2x2)**。
(II) 我们可以手动验证第一个输出值 **19** 的计算：
    --- (A) $0 \times 0 + 1 \times 1 + 3 \times 2 + 4 \times 3 = 19$
(III) d2l 的代码实现（`corr2d` 函数）验证了所有四个输出值：`[[19., 25.], [37., 43.]]`。

4.  **输出大小的计算（公式 7.2.2）**
(I) 输出张的大小通常**小于**输入张量。
(II) 假设输入大小为 $n_h \times n_w$，卷积核大小为 $k_h \times k_w$。
(III) 输出大小 $o_h \times o_w$ 计算如下：
    --- (A) $o_h = n_h - k_h + 1$
    --- (B) $o_w = n_w - k_w + 1$
(IV) 这个问题（尺寸变小）可以通过后续将学习的“填充”（padding）来解决。

5.  **corr2d 函数的实现**
(I) d2l 实现了 `corr2d` 函数来执行此操作。
```python
import torch
def corr2d(X, K): #@save
    """Compute 2D cross-correlation."""
    # 1. 获取卷积核的高度 (h) 和宽度 (w)
    h, w = K.shape
    
    # 2. 初始化输出张量 Y，形状根据 (n_h - k_h + 1) x (n_w - k_w + 1) 计算
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    
    # 3. 遍历输出张量 Y 的每一个元素 (i, j)
    # i 是行索引（高度），j 是列索引（宽度）
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # 4. 提取输入 X 中对应的窗口 (window)
            # X[i:i + h, j:j + w] 会切片出一个和 K 同样大小的区域
            window = X[i:i + h, j:j + w]
            
            # 5. 将窗口与卷积核 K 进行逐元素乘法，然后求和
            # 这就是互相关运算的核心
            Y[i, j] = (window * K).sum()
            
    return Y
# --- 验证 ---
# 定义输入张量 X (3x3)
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
# 定义卷积核 K (2x2)
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

# 执行互相关运算
output = corr2d(X, K)
print(output)
# tensor([[19., 25.],
#         [37., 43.]])
```

### 7.2.2 Convolutional Layers(卷积层)

1.  **卷积层的定义**
(I) 一个卷积层对其输入和卷积核执行**互相关**（cross-correlation）运算，并**添加一个标量偏置**（scalar bias），以此产生输出。
(II) 卷积层的两个可学习参数是：
    --- (A) **卷积核**（Kernel）（即权重$W$）。
    --- (B) **标量偏置**（Scalar Bias--$b$）。
(III) 就像全连接层一样，卷积核的参数通常也是**随机初始化**的。

2.  **`Conv2D` 类的实现**
一般来说:Layer 的实现包括:W,b,forward function.

```python
import torch
from torch import nn
# (这里假设我们已经定义了上一节的 corr2d 函数)
# def corr2d(X, K):
#     ... (省略 corr2d 的实现)

# 1. 定义一个二维卷积层类，它继承自 nn.Module
class Conv2D(nn.Module):
    
    # 2. 构造函数
    def __init__(self, kernel_size):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        
        # 3. 注册可学习的“权重”（即卷积核 K）
        # nn.Parameter() 会告诉 PyTorch 这是一个需要计算梯度的参数
        self.weight = nn.Parameter(torch.rand(kernel_size))
        
        # 4. 注册可学习的“偏置”（即标量 b）
        # 同样注册为 nn.Parameter
        self.bias = nn.Parameter(torch.zeros(1))

    # 5. 定义前向传播函数
    # 当调用 layer(x) 时，这个函数会自动执行
    def forward(self, x):
        # 6. 执行核心运算：互相关 + 偏置
        #    corr2d(x, K) + b
        return corr2d(x, self.weight) + self.bias
```

3.  **术语**
(I) 当我们说一个 $h \times w$ 的卷积层（或 $h \times w$ 卷积核）时，我们指的是该卷积层的**卷积核**高度为 $h$，宽度为 $w$。

### 7.2.3 Object Edge Detection in Images
1.  **卷积的一个简单应用**
(I) 目标：通过找到**像素变化**（pixel change）的位置来**检测图像中的边缘**。

2.  **构造“图像” X**

3.  **构造卷积核 K (边缘检测器)**
(I) 我们构造一个 1x2 的卷积核 `K`：
    --- (A) `K = torch.tensor([[1.0, -1.0]])`
(II) **核的功能**：
    --- (A) 这是一个**有限差分算子**（finite difference operator）。
    --- (B) 它在数学上是水平方向**一阶导数**的离散近似。
    --- (C) 当它“滑动”时，如果水平相邻的元素**相同**（如 `[1, 1]` 或 `[0, 0]`），输出为 $0$。
    --- (D) 如果水平相邻的元素**不同**（如 `[1, 0]` 或 `[0, 1]`），输出为**非零**。

4.  **执行互相关 (corr2d) 及结果**

```python
import torch
#--- 1. 定义 7.2.1 节中的核心函数 ---
#def corr2d(X, K): 

# --- 2. 构造 7.2.3 节的“图像” X ---
# 创建一个 6x8 的全 1 (白色) 张量
X = torch.ones((6, 8))

# 将中间 4 列 (索引 2 到 5) 设置为 0 (黑色)
# X[:, 2:6] 表示所有行，第2, 3, 4, 5列
X[:, 2:6] = 0

# 打印 X，可以看到一个带有垂直边缘的图像
# tensor([[1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.],
#         [1., 1., 0., 0., 0., 0., 1., 1.]])

# --- 3. 构造边缘检测卷积核 K ---
# K 是一个 1x2 的张量，用于检测水平方向的像素差异
# 这是一个有限差分算子
K = torch.tensor([[1.0, -1.0]])

# --- 4. 应用卷积核检测垂直边缘 ---
# 对 X 和 K 执行互相关运算
Y = corr2d(X, K)
# 打印 Y
# tensor([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#         [ 0.,  1.,  0.,  0.,  0., -1.,  0.]])
# 结果：Y 在第 1 列为 1 (检测到 1->0 边缘)，在第 5 列为 -1 (检测到 0->1 边缘)

# --- 5. 验证此卷积核无法检测水平边缘 ---
# X.t() 是 X 的转置，现在图像只有水平边缘
# X.t() 的形状是 8x6
X_transposed = X.t()
# 再次应用 1x2 的卷积核 K
Y_transposed = corr2d(X_transposed, K)
# 打印结果
# tensor([[0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0.]])
# 结果：输出全为 0，因为这个核 K 无法检测水平方向的边缘
```
5.  **此卷积核的局限性**：
    --- (A) 这个 `K` **只能**检测**垂直**边缘（即水平方向的像素变化）。
    --- (B) 当我们将这个核应用于 `X.t()`（转置后的图像，现在只有**水平**边缘）时，`corr2d(X.t(), K)` 的输出**全为零**。


### 7.2.4 Learning a Kernel 

1.  **手动设计核的局限性**
(I) 虽然为边缘检测手动设计一个 `[1, -1]` 的核是可行的，但当核更大、层数更深时，我们**不可能**手动指定所有核的功能。

2.  **新目标：从数据中学习核**
(I) 我们可以只看**输入-输出对**（`X` 和 `Y`），让模型**反向学习**出这个卷积核。
(II) **训练方法**：
    --- (A) 构造一个卷积层（`nn.LazyConv2d`），它的核（`weight`）被**随机初始化**。
    --- (B) 使用**平方误差**（`l = (Y_hat - Y) ** 2`）作为损失函数，比较该层的预测输出 `Y_hat` 和我们的目标 `Y`。
    --- (C) 对损失求和并进行**反向传播**（`l.sum().backward()`）来计算梯度。
    --- (D) 使用梯度下降法**手动更新**卷积核的权重：`conv2d.weight.data[:] -= lr * conv2d.weight.grad`。

3.  **实验设置**
(I) **模型**：`conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)`
    --- (A) `1` 个输出通道，核大小为 `(1, 2)`，无偏置。
(II) **输入 `X`**：重塑（`reshape`）为 `(1, 1, 6, 8)`。
    --- (A) 这是 PyTorch 卷积层要求的四维格式：（批量大小, 通道数, 高度, 宽度）。
(III) **目标 `Y`**：重塑为 `(1, 1, 6, 7)`。
(IV) **学习率 `lr`**：`3e-2` (即 0.03)。

4.  **结果**
(I) 训练循环（`for i in range(10):`）显示，`loss` 在 10 个 `epoch` 后从 16.481 迅速下降到了 0.274。
(II) 训练结束后，我们查看学习到的核：
    --- (A) `conv2d.weight.data.reshape((1, 2))`
(III) **学到的核**为：`tensor([[ 1.0398, -0.9328]])`。
(IV) 这个结果与我们之前手动定义的核 `K = [[1.0, -1.0]]` **惊人地接近**（remarkably close）。

5. **代码实现**

```python
import torch
import torch.nn as nn

# --- 1. 准备数据 (省略 7.2.1 和 7.2.3 前半部分的详细定义) ---

# a. 假设我们已定义 6x8 的输入图像 X，并塑形为 4D 张量
# 格式: (批量batch=1, 通道channel=1, 高height=6, 宽width=8)
X_true = torch.ones((6, 8))
X_true[:, 2:6] = 0
X = X_true.reshape((1, 1, 6, 8))

# b. 假设我们已定义 6x7 的目标输出 Y (即用 [1, -1] 核计算的结果)
# 格式: (批量batch=1, 通道channel=1, 高height=6, 宽width=7)
Y_true = torch.zeros((6, 7))
Y_true[:, 1] = 1.0  # 对应 [1, 0] 边缘
Y_true[:, 5] = -1.0 # 对应 [0, 1] 边缘
Y = Y_true.reshape((1, 1, 6, 7))


# --- 2. 构造并训练卷积层 (你提供的图片中的代码) ---

# 构造一个二维卷积层 (1个输出通道, 1x2核, 无偏置)
# 使用 LazyConv2d 来自动推断输入通道数
conv2d = nn.LazyConv2d(1, kernel_size=(1, 2), bias=False)

# 设置学习率
lr = 3e-2  # 0.03

# 开始 10 轮迭代的训练循环
for i in range(10):
    
    # 前向传播：计算预测值 Y_hat
    Y_hat = conv2d(X)
    
    # 计算损失：使用平方误差
    l = (Y_hat - Y) ** 2
    
    # 反向传播前，将梯度清零
    conv2d.zero_grad()
    
    # 计算梯度
    l.sum().backward()
    
    # 手动更新权重 (不使用优化器)
    # W = W - lr * W.grad
    conv2d.weight.data[:] -= lr * conv2d.weight.grad
    
    # 每 2 个 epoch 打印一次 loss
    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {l.sum():.3f}')

# --- 3. 检查学习到的核 (你提供的图片中的代码) ---
print("\n--- 训练完成 ---")

# 提取模型中的权重 (即卷积核) 并重塑为 1x2
learned_kernel = conv2d.weight.data.reshape((1, 2))

print(f"我们学习到的核: {learned_kernel}")

# 预期输出:
# epoch 2, loss 16.481
# epoch 4, loss 5.069
# epoch 6, loss 1.794
# epoch 8, loss 0.688
# epoch 10, loss 0.274
#
# --- 训练完成 ---
# 我们学习到的核: tensor([[ 1.0398, -0.9328]]) 
# (你的随机初始化不同，结果可能略有不同，但会很接近 [1.0, -1.0])
```


### 7.2.5 Cross-Correlation and Convolution

1.  **回顾：互相关 vs. 严格卷积**
(I) 我们再次回顾 7.1 节中的观察：深度学习中的“卷积”层（执行互相关）和数学上的“严格卷积”之间的对应关系。
(II) 严格的数学卷积（如 7.1.6 中定义）包含一个“翻转”操作。

2.  **如何用“互相关”实现“严格卷积”**
(I) 如果我们想得到“严格卷积”的输出，我们**只需要**做一件事：
    --- (A) 将我们的卷积核 `K` **水平和垂直地翻转**（flip）。
    --- (B) 用这个**翻转后的核**，去和输入 `X` 执行标准的**互相关**运算。

3.  **为什么在深度学习中这个区别不重要？**
(I) 核心原因：**卷积核 K 是从数据中学习到的**。
(II) 无论框架在底层是执行“互相关”还是“严格卷积”，网络的输出**都不会受影响**。
(III) **一个思想实验**：
    --- (A) 假设一个层用“互相关”学习到了最优核 `K`。
    --- (B) 另一个层用“严格卷积”学习，它最终会学习到一个最优核 `K'`。
    --- (C) 结论：为了产生相同的输出，`K'` 必然会是 `K` 的**翻转版本**（`K' = flip(K)`）。
(IV) 既然学习算法会自动“吸收”这个翻转操作，我们就没必要区分这两种运算了。

4.  **术语约定**
(I) 遵循深度学习文献的标准术语，我们将**继续**把“互相关运算”称为“**卷积**”。
(II) 尽管从数学上讲，它们（`cross-correlation` 和 `convolution`）略有不同。


### 7.2.6 Feature Map and Receptive Field(特征图与感受野)
1.  **特征图 (Feature Map)**
(I) 正如 7.1.4 节所述，卷积层的输出有时也被称为**特征图**。
(II) 它可以被视为（在空间维度上）学习到的、传递给后续层的**特征**（representations / features）。

2.  **感受野 (Receptive Field)**
(I) 在 CNN 中，对于某一层中的任何元素 `x`，其**感受野**（receptive field）指的是**所有**能够影响 `x` 在前向传播中计算的、来自**先前层**的元素。
(II) 注意：感受野可能大于输入的实际大小（例如，当层数很深时）。

3.  **感受野的例子与重要性**
(I) **单层**：
    --- (A) 一个 2x2 的卷积核，其输出元素的感受野，就是输入中的 2x2 区域。
(II) **堆叠层（Deeper CNN）**：
    --- (A) 假设我们有一个 2x2 的输出 `Y`（来自第一层）。
    --- (B) 我们再用一个 2x2 的核对 `Y` 进行卷积，产生一个**单一**的输出元素 `z`。
    --- (C) `z` 在 `Y` 上的感受野是 `Y` 的**全部** 4 个元素。
    --- (D) `z` 在**原始输入**（`X`）上的感受野是**全部** 9 个元素（3x3）。
(III) **核心思想**：当我们需要一个更大的感受野来检测更广泛的输入特征时，我们可以**构建更深的网络**（build a deeper network）。

4.  **生物学起源**
(I) “感受野”这个名字来源于**神经生理学**（neurophysiology）。
(II) Hubel 和 Wiesel 的一系列实验（1959-1968）探索了动物**视觉皮层**（visual cortex）的反应。
(III) 他们发现，较低层次的神经元会对**边缘**（edges）和相关形状作出反应。
(IV) 这种相似性甚至在现代图像分类任务中训练的**深度网络**所计算出的特征中仍然存在。

5.  **总结**
(I) 卷积已被证明是计算机视觉中一个极其强大的工具，无论是在**生物学**中还是在**代码**中。
(II) 因此（事后看来），它们预示着深度学习近期的成功也就不足为奇了。


### 7.2.7 Summary (总结)

1.  **卷积层的核心计算**
(I) 卷积层所需的核心计算是**互相关**（cross-correlation）运算。
(II) 我们看到，一个简单的**嵌套 for 循环**（nested for-loop）就足以计算它。
(III) 当存在多输入和多输出通道时，它表现为通道间的**矩阵-矩阵运算**。

2.  **局部性 (Locality) 及其优势**
(I) 卷积计算是直接了当的，并且最重要的是，它是**高度局部**（highly local）的。
(II) 这种特性允许进行显著的**硬件优化**。
(III) 芯片设计者可以专注于**快速计算**（fast computation）而不是内存。
(IV) 这为普及且廉价的计算机视觉（ubiquitous and affordable computer vision）打开了大门。

3.  **卷积核的功能与学习**
(I) 卷积本身可以用于多种目的：
    --- (A) 检测边缘和线条。
    --- (B) 模糊图像（blurring）。
    --- (C) 锐化图像（sharpening）。
(II) **最重要的一点**：统计学家（或工程师）**不必**去“发明”合适的滤波器。
(III) 相反，我们可以**从数据中学习**（learn them from data）它们。
(IV) 这用“基于证据的统计”取代了“特征工程的启发式方法”。

4.  **与生物学的联系**
(I) 这些滤波器不仅对构建深度网络有优势，它们还对应于大脑中的**感受野**（receptive fields）和**特征图**（feature maps）。
(II) 这给了我们信心，表明我们正走在正确的轨道上。
---
## 7.3 Padding and Stride(填充与步幅)

1.  **卷积运算的问题：尺寸缩小**
(I) 我们回顾 7.2.1 节中的例子：输入 $3 \times 3$，卷积核 $2 \times 2$，输出 $2 \times 2$。
(II) 输出形状的通用公式为：$(n_h - k_h + 1) \times (n_w - k_w + 1)$。
(III) **动机（Motivation）**：
    --- (A) 当我们应用**许多连续的**卷积层时，输出会变得比输入**小得多**。
    --- (B) **一个例子**：如果一个 $240 \times 240$ 的图像，经过10层 $5 \times 5$ 的卷积，图像会缩小到 $200 \times 200$ 像素。
    --- (C) 这不仅损失了 30% 的图像面积，还**抹去**（obliterating）了原始图像**边界**（boundaries）上的信息。

2.  **控制输出大小的新技术**
(I) **填充（Padding）**：
    --- (A) 这是解决上述（尺寸缩小）问题**最流行**的工具。
(II) **步幅卷积（Strided Convolutions）**：
    --- (A) 在其他情况下，我们可能**希望**（want）大幅降低维度（例如，当原始输入分辨率过高时）。
    --- (B) 步幅卷积是实现这一目标的流行技术。

```python
import torch
from torch import nn
```

### 7.3.1 Padding

1.  **填充的动机 (Motivation)**
(I) 卷积层的一个棘手问题是，我们往往会**丢失**（lose）图像**周界**（perimeter）的像素。
(II) 图像**角落**（corners）的像素几乎没有被使用。
(III) 这个问题在连续（successive）的卷积层中会**累积**，导致输出迅速缩小。

2.  **填充的定义**
(I) 一个直接的解决方案是在输入图像的**边界**（boundary）周围**添加额外的像素**（extra pixels of filler）。
(II) 这被称为**填充**（Padding）。
(III) 通常，我们将这些额外像素的值设置为**零**。

3.  **填充后的输出形状 (公式 7.3.1)**
(I) 假设我们总共添加了 $p_h$ 行填充（通常一半在顶部，一半在底部）和 $p_w$ 列填充（通常一半在左侧，一半在右侧）。
(II) 输出形状将变为：
    $$(n_h - k_h + p_h + 1) \times (n_w - k_w + p_w + 1)$$
(III) 这意味着输出的高度和宽度将分别增加 $p_h$ 和 $p_w$。

4.  **目标：保持维度不变**
(I) 在许多情况下，我们希望**输入和输出具有相同的高度和宽度**。
(II) 这使得预测网络中每一层的输出形状变得更容易。
(III) 要实现这一点，我们只需根据公式设置：
    --- (A) $p_h = k_h - 1$
    --- (B) $p_w = k_w - 1$

5.  **奇数卷积核的好处**
(I) CNNs 通常使用**奇数**（odd）高度和宽度的卷积核（例如 1, 3, 5, 7）。
(II) **好处**：这使我们能够通过在顶部/底部和左侧/右侧填充**相同数量**的行/列来保持维度。
(III) **示例**：如果 $k_h = 3$ (奇数)，我们设置 $p_h = 2$。在 PyTorch 中，我们可以通过 `padding=1` 来实现，它会在顶部和底部**各**添加 1 行。
(IV) **代码验证 (kernel=3, padding=1)**：
```python
import torch
from torch import nn

# 我们定义一个辅助函数来计算卷积。
# 它初始化卷积层权重，并对输入和输出执行相应的维度“提升”和“缩减”
def comp_conv2d(conv2d, X):
    
    # (1, 1) 表示批量大小和通道数都是 1
    # X.reshape((1, 1) + X.shape) 将 2D 输入 X (h, w) 变为 4D (1, 1, h, w)
    X = X.reshape((1, 1) + X.shape)
    
    # 将 4D 张量 X 传入卷积层
    Y = conv2d(X)
    
    # 剥离 (Strip) 前两个维度：批量和通道
    # Y.reshape(Y.shape[2:]) 将 4D 输出 (1, 1, h, w) 变回 2D (h, w)
    return Y.reshape(Y.shape[2:])

# --- 示例：使用 k_h=3, k_w=3, padding=1 ---

# 在每条边填充 1 行或 1 列，总共添加 2 行或 2 列
# (p_h = 1+1 = 2; p_w = 1+1 = 2)
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)

# 创建一个 8x8 的 2D 随机输入张量
X = torch.rand(size=(8, 8))

# 调用辅助函数计算卷积，并获取输出的形状
comp_conv2d(conv2d, X).shape
# 预期输出: torch.Size([8, 8])
# (因为 n_h-k_h+p_h+1 = 8-3+2+1 = 8)
```

--- (A) `conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)`
--- (B) `X` 是一个 8x8 的随机输入。
--- (C) `comp_conv2d(conv2d, X).shape` 的输出是 `torch.Size([8, 8])`，维度被**保持**。
(V) 这种“奇数核 + 对称填充”的组合有一个**额外的好处**：输出 `Y[i, j]` 是由以输入 `X[i, j]` 为中心的窗口计算得出的。

6.  **不同核大小与填充**
(I) 当卷积核的高度和宽度不同时，我们可以通过设置**不同**的填充数来使输出和输入具有相同的高度和宽度。
(II) **代码验证 (kernel=(5, 3), padding=(2, 1))**：
```python
# We use a convolution kernel with height 5 and width 3. 
# The padding on either side of the height and width are 2 and 1
conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape #torch.Size([8, 8])
```
--- (A) `conv2d = nn.LazyConv2d(1, kernel_size=(5, 3), padding=(2, 1))`
--- (B) `kernel_size=(5, 3)` 表示 $k_h=5, k_w=3$。
--- (C) `padding=(2, 1)` 表示在高度上**各**填充 2 (总 $p_h=4$)，在宽度上**各**填充 1 (总 $p_w=2$)。
--- (D) 这完美符合了 $p_h = k_h - 1$ ($4 = 5 - 1$) 和 $p_w = k_w - 1$ ($2 = 3 - 1$) 的要求。
--- (E) 8x8 的输入，输出同样是 `torch.Size([8, 8])`。


### 7.3.2 Stride(步幅)

1.  **步幅的定义**
(I) 在前面的例子中，我们的卷积窗口默认**每次滑动一个元素**（即步幅为1）。
(II) **步幅**（Stride）是指卷积窗口**每次滑动**（slide）所**遍历**（traversed）的行数和列数。
(III) 为什么要使用更大的步幅？
--- (A) 为了**计算效率**（computational efficiency）。
--- (B) 当我们希望对图像进行**下采样**（downsample）时。
--- (C) 当卷积核很大时，这尤其有用，因为它仍然可以捕获大范围的图像信息。

2.  **步幅的计算示例**
(I) 例子:设高度（垂直）步幅为 $s_h=3$，宽度（水平）步幅为 $s_w=2$。
(II) 第一个输出 `0`（原文为8）是通过计算左上角得到的。
(III) 第二个**列**元素 `6` 是通过将窗口**向下滑动 3 行**（$s_h=3$）计算得到的（`0*0+6*1+0*2+0*3=6`）。
(IV) 第二个**行**元素 `8` 是通过将窗口**向右滑动 2 列**（$s_w=2$）计算得到的。

3.  **步幅的输出形状 (公式 7.3.2)**
(I) 在一般情况下，当高度步幅为 $s_h$，宽度步幅为 $s_w$ 时，输出形状为：
    $$\lfloor(n_h - k_h + p_h + s_h) / s_h\rfloor \times \lfloor(n_w - k_w + p_w + s_w) / s_w\rfloor$$
(II) **一个有用的简化**：
--- (A) 如果我们设置 $p_h = k_h - 1$ 和 $p_w = k_w - 1$（即在 $s=1$ 时保持维度的填充），则公式简化为 $\lfloor(n_h + s_h - 1) / s_h\rfloor \times \lfloor(n_w + s_w - 1) / s_w\rfloor$。
(III) **另一个简化**：
--- (A) 如果输入的高度和宽度**可以被步幅整除**，则输出形状为 $(n_h / s_h) \times (n_w / s_w)$。

4.  **代码示例**
```python
X = torch.rand(size=(8, 8))
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape #torch.Size([4, 4])

conv2d = nn.LazyConv2d(1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape #torch.Size([2, 2])
```

### 7.3.3 Summary and Discussion

1.  **填充 (Padding) 的作用**
(I) 填充可以增加输出的高度和宽度。
(II) 它的主要目的是为了让输出和输入具有**相同的高度和宽度**，以避免不希望的**收缩**（shrinkage）。
(III) 它可以确保所有像素被**同等频繁**地使用。
(IV) 通常我们使用对称填充（symmetric padding），如果 $p_h = p_w$，我们就简称为填充 $p$。

2.  **步幅 (Stride) 的约定**
(I) 类似的约定也适用于步幅：如果 $s_h = s_w$，我们就简称为步幅 $s$。
(II) 步幅可以**降低输出的分辨率**（例如，将高和宽降低到输入的 $1/n$）。
(III) **默认情况下，填充为 0，步幅为 1**。

3.  **零填充 (Zero-Padding) 的好处**
(I) 我们到目前为止讨论的都是用**零**扩展图像（零填充）。
(II) 这具有显著的**计算优势**（computational benefit）。
(III) **(优化)**：操作（Operators）可以被设计为**隐式**（implicitly）利用这种填充，而**无需分配额外的内存**。
(IV) **(特征)**：它允许 CNNs 通过学习“空白”（whitespace）的位置来编码**隐式的空间位置信息**。
(V) 存在零填充之外的替代方案，但除非出现“伪影”（artifacts），否则没有明确的理由使用非零填充。
---
## 7.4 Multiple Input and Multiple Output Channels

1.  **回顾与现状**
(I) 我们在 7.1.4 节中已经描述过彩色图像（如RGB）的多通道概念。
(II) 但是，到目前为止，我们所有的数值示例都**简化**（simplified）了，只使用**单个输入**和**单个输出**通道（即二维张量）。

2.  **引入通道后的变化**
(I) 当我们加入通道后，我们的**输入**和**隐藏表示**都变成了**三维张量**。
(II) 例如，一个 RGB 输入图像的形状是 $3 \times h \times w$。
(III) 我们将这个大小为 3 的轴称为**通道维度**（channel dimension）。

3.  **本节目标**
(I) 通道这个概念和 CNN 本身一样古老（例如 LeNet-5 就在使用）。
(II) 在本节中，我们将**深入**（deeper look）研究具有**多输入**和**多输出**通道的卷积核。

### 7.4.1 Multiple Input Channels

1.  **问题与核的形状**
(I) 当输入数据包含多个通道 (channels) 时（例如 $c_i$ 个通道），我们需要构造一个卷积核。
(II) 这个卷积核**必须**具有与输入数据**相同数量**的输入通道（即 $c_i$）。
(III) 因此，如果核的窗口形状是 $k_h \times k_w$，那么完整的核张量形状是 $c_i \times k_h \times k_w$。
--- (A) 它包含一个 $k_h \times k_w$ 的张量，用于**每一个**（every）输入通道。

2.  **多输入通道的互相关运算**
(I) 运算过程如下：
--- (A) 在**每个**通道上，分别执行二维互相关运算（即输入的第 $i$ 个通道，与核的第 $i$ 个通道进行 `corr2d`）。
--- (B) 将 $c_i$ 个（例如，Fig. 7.4.1 中是2个）通道的运算结果**相加**（summing over the channels）。
--- (C) 最终产生一个**二维**的输出张量。
(II) **eg:（计算输出 56）**：
--- (A) **通道 1**：$(1\times 1 + 2\times 2 + 4\times 3 + 5\times 4) = 37$
--- (B) **通道 2**：$(0\times 0 + 1\times 1 + 3\times 2 + 4\times 3) = 19$
--- (C) **总和**：$37 + 19 = 56$

3.  **代码实现 `corr2d_multi_in`与代码验证**
``` python
import torch
# 假设 d2l 库已经导入，并提供了 d2l.corr2d 函数
from d2l import torch as d2l 

def corr2d_multi_in(X, K):
    """计算多输入通道的二维互相关。"""
    
    # X 和 K 都是 3D 张量 (c_in, h, w) 或 (c_in, k_h, k_w)
    # zip(X, K) 会将 X 的第i个通道 (x) 与 K 的第i个通道 (k) 配对
    # d2l.corr2d(x, k) 对每个通道对执行标准 2D 互相关
    # sum() 将所有通道的结果（2D矩阵）相加，得到最终的 2D 输出矩阵
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

# --- 验证 (对应eg) ---
# 1. 定义输入张量 X (shape: [2, 3, 3])
#    c_in=2, n_h=3, n_w=3
X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],  # 第 0 通道
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]) # 第 1 通道

# 2. 定义卷积核 K (shape: [2, 2, 2])
#    c_in=2, k_h=2, k_w=2
K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]],  # 对应第 0 通道的核
                  [[1.0, 2.0], [3.0, 4.0]]]) # 对应第 1 通道的核

# 3. 执行多输入通道互相关
#    (输出形状为 (n_h-k_h+1, n_w-k_w+1) = (2, 2))
corr2d_multi_in(X, K)
# 预期输出:
# tensor([[ 56.,  72.],
#         [104., 120.]])
```
### 7.4.2 Multiple Output Channels

1.  **多输出通道的动机**
(I) 无论有多少输入通道，到目前为止，我们总是只得到**一个**输出通道。
(II) 但在神经网络中，每一层都拥有**多个通道**是至关重要的。
(III) 在流行的网络架构中，我们倾向于在网络加深时**增加**（increase）通道维度，并通常通过**下采样**（down-sampling）来**牺牲空间分辨率**（spatial resolution）以换取**通道深度**（channel depth）。
(IV) 直观上，每个通道响应一组不同的特征，但实际上它们是**共同优化**（jointly useful）的。

2.  **多输出通道的卷积核**
(I) 设 $c_i$ 为输入通道数，$c_o$ 为输出通道数。
(II) 为了得到 $c_o$ 个输出，我们必须为**每一个**（for every）输出通道 `d` 准备一个形状为 $c_i \times k_h \times k_w$ 的 3D 卷积核。
(III) 我们将这 $c_o$ 个 3D 卷积核**堆叠**（concatenate）在一起。
(IV) 最终，完整的卷积核 `K` 是一个 **4D 张量**，形状为：$c_o \times c_i \times k_h \times k_w$。

3.  **代码实现 `corr2d_multi_in_out`**

```python
def corr2d_multi_in_out(X, K):
    """计算多输入和多输出通道的二维互相关。"""
    
    # K 是一个 4D 张量 (c_o, c_i, k_h, k_w)
    # X 是一个 3D 张量 (c_i, n_h, n_w)
    # 1. 迭代 K 的第 0 维 (输出通道 c_o)
    #    在每次循环中, 'k' 是一个 3D 张量 (c_i, k_h, k_w)
    # 2. 对每个 3D 核 'k' 和 3D 输入 'X' 调用 corr2d_multi_in,
    #    这会计算出一个 2D 的输出特征图 (o_h, o_w)
    # 3. torch.stack 将所有 c_o 个 2D 特征图堆叠在一起
    #    dim=0 指定沿着新的第 0 维堆叠
    # 4. 最终返回一个 3D 张量 (c_o, o_h, o_w)
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)
```

4.  **代码验证**
```python
K = torch.stack((K, K + 1, K + 2), 0)#构造新的4维张量K
K.shape #torch.Size([3, 2, 2, 2])

output = corr2d_multi_in_out(X, K)
print(output)
# 预期输出:
# tensor([[[ 56.,  72.],
#          [104., 120.]],   # <-- 这是 K 计算的
#
#         [[ 76., 100.],
#          [148., 172.]],   # <-- 这是 K+1 计算的
#
#         [[ 96., 128.],
#          [192., 224.]]])  # <-- 这是 K+2 计算的
```

### 7.4.3 $1\times1$ Convolutional Layer


1.  **1x1 卷积的初见**
(I) 乍一看， $k_h = k_w = 1$ 的 1x1 卷积似乎没有意义，因为它**不关联（correlates）** 相邻像素。
(II) 尽管如此，它们是复杂深度网络设计中的流行操作。

2.  **1x1 卷积的真正作用**
(I) 1x1 卷积**失去**了识别空间模式（高和宽维度）的能力。
(II) 它的**唯一计算**发生在**通道维度**（channel dimension）上。
(III) 输入和输出具有**相同的高度和宽度**。
(IV) **核心思想**：
--- (A) 输出中的每个元素，都是从输入中**相同位置**（at the same position）的元素的**线性组合**（linear combination）派生出来的。
--- (B) 我们可以将其视为一个**全连接层**（fully connected layer），它被应用于**每一个**（every single）像素位置。
--- (C) 在每个位置上，它都将 $c_i$ 个输入值（来自输入通道）转换为 $c_o$ 个输出值（变为输出通道）。
(V) **(参数共享)**：因为它仍然是一个“卷积”，所以这个全连接层的权重（$c_o \times c_i$ 个）在所有像素位置上是**共享的**（tied across pixel location）。
(VI) **(非线性)**：1x1 卷积层通常后跟**非线性激活函数**。
--- (A) 这至关重要，因为它确保 1x1 卷积**不能**被“折叠”（folded）或（在数学上）合并到其他卷积层中。

3.  **用全连接层实现 1x1 卷积**
```python
def corr2d_multi_in_out_1x1(X, K):
    """
    使用矩阵乘法 (全连接层) 来实现 1x1 卷积。
    参数:
    X: 3D 输入张量 (c_i, h, w)
    K: 4D 卷积核张量 (c_o, c_i, 1, 1)
    """
    # 1. 获取输入的形状
    # c_i = 输入通道数, h = 高, w = 宽
    c_i, h, w = X.shape
    
    # 2. 获取输出通道数
    # K 的形状是 (c_o, c_i, 1, 1)，所以 K.shape[0] 就是 c_o
    c_o = K.shape[0]
    
    # 3. 重塑 (Reshape) 输入 X
    # 将 X 从 (c_i, h, w) 变为 (c_i, h * w)
    # 这等价于：将 (h, w) 个像素“展平”，
    # 形成一个包含 (h * w) 个向量的“批量”，
    # 每个向量的维度是 c_i (输入通道数)。
    X = X.reshape((c_i, h * w))
    
    # 4. 重塑 (Reshape) 卷积核 K
    # 将 K 从 (c_o, c_i, 1, 1) 变为 (c_o, c_i)
    # 这就是全连接层的权重矩阵 W 的形状，
    # 它将 c_i 个输入特征映射到 c_o 个输出特征。
    K = K.reshape((c_o, c_i))
    
    # 5. 执行矩阵乘法 (全连接层的核心)
    # Y = W * X
    #   K 的形状: (c_o, c_i)
    #   X 的形状: (c_i, h * w)
    #   Y 的形状: (c_o, h * w)
    Y = torch.matmul(K, X)
    
    # 6. 重塑 (Reshape) 输出 Y
    # 将 Y 从 (c_o, h * w) 恢复为 (c_o, h, w)
    # 这就还原了卷积输出应有的 3D 形状 (多通道特征图)
    return Y.reshape((c_o, h, w))
```

4.  **验证等效性**
```python
# a. 创建一个 3D 随机输入 X
#    形状: (c_i=3, h=3, w=3)
X = torch.normal(0, 1, (3, 3, 3))

# b. 创建一个 4D 随机 1x1 卷积核 K
#    形状: (c_o=2, c_i=3, k_h=1, k_w=1)
K = torch.normal(0, 1, (2, 3, 1, 1))

# c. 计算 Y1 (使用 1x1 的矩阵乘法实现)
Y1 = corr2d_multi_in_out_1x1(X, K)

# d. 计算 Y2 (使用通用的互相关实现)
Y2 = corr2d_multi_in_out(X, K)

# e. 断言 (Assert)：
#    检查 Y1 和 Y2 之间的绝对差值总和是否小于 1e-6 (0.000001)
#    torch.abs(Y1 - Y2) 计算逐元素的绝对差值
#    .sum() 计算所有差值的总和
#    float(...) 将结果转为 Python 浮点数
#    assert ... < 1e-6 确保这个差值非常小，证明两者等效
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6
```

### 7.4.4 Discussion

1.  **通道 (Channels) 的好处**
(I) 通道允许我们**结合两者(MLP+convolution)的优点**（combine the best of both worlds）：
--- (A) **MLP** 的优点（允许显著的**非线性**，例如 1x1 卷积后跟随一个非线性激活函数）。
--- (B) **卷积**的优点（允许对特征进行**局部化分析**）。
(II) CNNs 可以**同时推理多种特征**（例如，边缘检测器和形状检测器）。
(III) 通道提供了一个实用的**权衡**（trade-off）：
--- (A) 它介于“平移不变性”和“局部性”带来的**参数大幅减少**。
--- (B) 通道 (Channels) 是我们用来在 **“模型效率”和“模型能力”** 之间找到完美平衡点的工具。

2.  **灵活性的代价：计算成本**
(I) 这种灵活性是有代价的（comes at a price）。
(II) **计算成本**：对于 $h \times w$ 的图像，$k \times k$ 的核，$c_i$ 个输入通道和 $c_o$ 个输出通道，其成本为：
--- (A) $O(h \cdot w \cdot k^2 \cdot c_i \cdot c_o)$
(III) **一个例子**：
--- (A) $256 \times 256$ 图像，$5 \times 5$ 核，$c_i=128$, $c_o=128$。
--- (B) 这将导致**超过 530 亿**（53 billion）次操作。

3.  **展望：降低成本的策略**
(I) 我们后续将学习降低此成本的有效策略。
(II) 例如，要求通道间的操作是**块对角**（block-diagonal）的。
(III) 这催生了像 **ResNeXt** 这样的架构。
---

## 7.5 Pooling(池化)

1.  **池化的动机 (Motivation)**
(I) **动机一：获取全局表示（下采样）**
--- (A) 我们的最终任务通常是一个**全局问题**（例如，“图像中是否包含一只猫？”）。
--- (B) 这要求网络的最终输出对**整个输入**都敏感。
--- (C) 我们通过 **“逐步聚合信息”** ，产生“越来越粗糙的特征图”（coarser and coarser maps）来实现这一目标。
--- (D) **降低空间分辨率**（Reducing spatial resolution）会加速这个过程（即“空间下采样”），因为卷积核可以覆盖更大的有效区域。
(II) **动机二：位置敏感性（平移不变性）**
--- (A) 卷积层（如 7.2 节的边缘检测器）对特征的**位置**（location）非常**敏感**。
--- (B) **一个例子**：如果我们将图像 `X` 平移 1 个像素得到 `Z`（`Z[i, j] = X[i, j + 1]`），那么新图像 `Z` 的输出可能会与 `X` 的输出**截然不同**（vastly different）。
--- (C) 这是一个问题，因为在现实中，物体几乎**不会**出现在完全相同的位置（例如，相机振动）。

2.  **池化层的定义**
(I) 本节介绍**池化层**（pooling layers）。
(II) 池化层服务于**双重目的**（dual purposes）：
--- (A) 缓解（mitigating）卷积层对**位置**的**敏感性**。
--- (B) 对表示（representations）进行**空间下采样**（spatially downsampling）。

```python
import torch
from torch import nn
from d2l import torch as d2l
```

### 7.5.1 Maximum Pooling and Average Pooling

1.  **池化算子 (Pooling Operators) 的定义**
(I) 池化算子由一个**固定形状的窗口**（`pool_size`，称为**池化窗口**）组成，它会“滑动”遍历输入的所有区域（默认步幅为1）。
(II) **与卷积的关键区别**：池化层**不包含参数**（`no parameters`），即它**没有可学习的卷积核**（`no kernel`）。
(III) 池化算子是**确定性的**（deterministic），它们只是计算池化窗口中元素的**最大值**（`max`）或**平均值**（`average`）。
(IV) 这两种操作分别称为**最大池化**（max-pooling）和**平均池化**（average pooling）。

2.  **Max-pooling vs. Avg-pooling**
(I) **平均池化**：与 CNNs 一样古老。
--- (A) 类似于下采样，但它通过对相邻像素取**平均**来获得更好的**信噪比**（signal-to-noise ratio）。
(II) **最大池化**：
--- (A) 在几乎所有情况下，最大池化都**优于**（preferable to）平均池化。

3.  **池化(pooling)代码**
```python
import torch

def pool2d(X, pool_size, mode='max'):
    """
    计算二维池化 (最大池化或平均池化)。
    参数:
    X:         输入张量 (2D)
    pool_size: 池化窗口的大小 (p_h, p_w)
    mode:      'max' 或 'avg'
    """
    # 1. 获取池化窗口的高度 (p_h) 和宽度 (p_w)
    p_h, p_w = pool_size
    # 2. 初始化输出张量 Y
    #    (这里假设步幅 stride=1)
    #    输出形状: (n_h - p_h + 1) x (n_w - p_w + 1)
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    # 3. 遍历输出张量 Y 的所有坐标 (i, j)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            
            # 4. 提取输入 X 中对应的窗口
            window = X[i: i + p_h, j: j + p_w]
            
            # 5. 根据 'mode' 执行计算
            if mode == 'max':
                # 计算该窗口内的最大值
                Y[i, j] = window.max()
            elif mode == 'avg':
                # 计算该窗口内的平均值
                Y[i, j] = window.mean()
                
    return Y
# --- 验证 ---
# 1. 定义输入张量 X (来自 Fig 7.5.1)
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

# 2. 验证最大池化 (Max-Pooling)
#    (mode='max' 是默认值)
max_pool_output = pool2d(X, (2, 2))
print(max_pool_output)
# 预期输出 :
# tensor([[4., 5.],
#         [7., 8.]])

# 3. 验证平均池化 (Average-Pooling)
avg_pool_output = pool2d(X, (2, 2), 'avg')
print(avg_pool_output)
# 预期输出 :
# tensor([[2., 3.],
#         [5., 6.]])
```
### 7.5.2 Padding and Stride

1.  **在池化层中使用填充和步幅**
(I) 与卷积层一样，池化层也会改变输出形状。
(II) 我们可以通过调整**填充**（padding）和**步幅**（stride）来控制输出形状。
(III) 我们将使用PyTorch内置的 `nn.MaxPool2d` 层来演示。

2.  **构造输入张量 X**
(I) 我们首先构造一个 4D 的输入张量 `X`（批量=1, 通道=1, 高=4, 宽=4）。

3.  **框架的默认步幅行为**
(I) 在深度学习框架中，池化层**默认**（by default）的**步幅**与**池化窗口**（pooling window）的大小**相匹配**。
(II) 注意：池化层没有模型参数，因此不需要初始化。

4.  **手动指定填充和步幅**
(I) 我们可以手动指定 `padding` 和 `stride` 来覆盖（override）默认值。
5.  **任意矩形窗口**
(I) 我们可以为窗口、步幅和填充指定**不同**的高度和宽度。

6. **代码实现**
```python
import torch
import torch.nn as nn
# 1. 创建 4D 示例输入 X (1, 1, 4, 4)，内容为 0~15
X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))
print(X)
# --- 示例 1：默认步幅 (stride=kernel_size) ---
# 定义 3x3 池化层 (PyTorch 默认 stride=3)
pool2d = nn.MaxPool2d(3)
# 池化层没有参数，无需初始化
pool2d(X)
# 预期输出: tensor([[[[10.]]]])


# --- 示例 2：手动指定填充和步幅 ---
# 定义 3x3 池化层, 填充为 1, 步幅为 2
pool2d = nn.MaxPool2d(3, padding=1, stride=2)
pool2d(X)
# 预期输出: tensor([[[[ 5.,  7.],
#                  [13., 15.]]]])


# --- 示例 3：矩形窗口和非对称参数 ---
# 定义 k=(2, 3), s=(2, 3), p=(0, 1) 的池化层
pool2d = nn.MaxPool2d((2, 3), stride=(2, 3), padding=(0, 1))
pool2d(X)
# 预期输出: tensor([[[[ 5.,  7.],
#                  [13., 15.]]]])
```

### 7.5.3  Multiple Channels(多通道)
1.  **池化层如何处理多通道**
(I) 当处理多通道输入数据时，池化层（`nn.MaxPool2d`）会**分别**（separately）池化**每一个**输入通道。
(II) **关键区别**：它**不会**像卷积层（`nn.Conv2d`）那样，将所有输入通道**相加**（summing up over channels）。
(III) **结论**：池化层的**输出通道数**（number of output channels）与**输入通道数**（number of input channels）**相同**。

2.  **构造多通道输入**
```python
# 假设 X 是 7.5.2 节中 (1, 1, 4, 4) 的张量
# 1. 构造一个 2 通道输入 X
#    torch.cat 在 dim=1 (通道) 维度上拼接 X 和 X+1
X = torch.cat((X, X + 1), 1)
print(X)
# 预期输出 (X):
# tensor([[[[ 0.,  1.,  2.,  3.],
#           [ 4.,  5.,  6.,  7.],
#           [ 8.,  9., 10., 11.],
#           [12., 13., 14., 15.]],
#
#          [[ 1.,  2.,  3.,  4.],
#           [ 5.,  6.,  7.,  8.],
#           [ 9., 10., 11., 12.],
#           [13., 14., 15., 16.]]]])
```

3.  **多通道池化验证**
```python
# 2. 定义池化层
#    (与 7.5.2 节中的示例 2 相同)
pool2d = nn.MaxPool2d(3, padding=1, stride=2)

# 3. 将池化层应用于 2 通道输入
print(pool2d(X))
# 预期输出 (pool2d(X)):
# tensor([[[[ 5.,  7.],
#           [13., 15.]],
#
#          [[ 6.,  8.],
#           [14., 16.]]]])
```
### 7.5.4 总结 (Summary)

1.  **池化 (Pooling) 的核心特性**
(I) 池化是一个**极其简单**（exceedingly simple）的操作：它如其名，在一个窗口上**聚合**（aggregate）结果。
(II) 所有卷积的“语义”（semantics），如**步幅**（strides）和**填充**（padding），都以相同的方式适用于池化。

2.  **池化与通道**
(I) 池化**对通道是“冷漠的”**（indifferent to channels）。
(II) 这意味着：
--- (A) 它**不会**改变通道的数量（`number of channels unchanged`）。
--- (B) 它**分别**（separately）应用于**每一个**通道。

3.  **池化的选择与目的**
(I) 在两种流行的选择中，**最大池化**（max-pooling）通常**优于**（preferable to）**平均池化**（average pooling）。
(II) 原因是最大池化为输出赋予了（confers）一定程度的**不变性**（invariance）。
(III) 一个流行的选择是使用 $2 \times 2$ 的池化窗口，这会使空间分辨率**减少为四分之一**（quarter the spatial resolution）。

4.  **超越池化 (Beyond Pooling)**
(I) 存在许多超越池化的降低分辨率的方法，例如：
--- (A) **随机池化**（stochastic pooling）。
--- (B) **分数最大池化**（fractional max-pooling）。
(II) 我们稍后将看到的**注意力机制**（attention mechanism）将提供更精细的（refined）聚合输出的方法。
---

## 7.6 Convolutional Neural Networks (LeNet)

1.  **回顾：从 MLP 到 CNN**
(I) 我们现在拥有了组装一个全功能 CNN 所需的所有组成成分（ingredients）。
(II) 在我们之前的尝试中（如 Section 4.4 和 5.2），我们必须将图像**展平**（`flattened`）为 784 维向量，以供线性模型或 MLP 处理。
(III) 现在，我们有了卷积层，我们可以**保留**（retain）图像的**空间结构**（spatial structure）。
(IV) 用卷积层替代全连接层，可以让我们得到**更精简的模型**（`parsimonious models`），即参数更少。

2.  **LeNet 的引入**
(I) LeNet 是最早在计算机视觉任务上引起广泛关注的 CNN 之一。
(II) 它由 Yann LeCun 在 AT&T 贝尔实验室引入，用于**识别手写数字**（recognizing handwritten digits）。
(III) 这项工作是十年研究的结晶；LeCun 的团队发表了第一项通过**反向传播**（backpropagation）成功训练 CNNs 的研究（LeCun et al., 1989）。

3.  **LeNet 的成就与遗产**
(I) LeNet 取得了与当时占主导地位的支持向量机（SVMs）相媲美的优异结果。
(II) 它在手写数字上实现了**低于 1%** 的每位数字错误率。
(III) LeNet 最终被应用于 **ATM 机**中处理存款。
(IV) 直到今天，一些 ATM 机仍在运行 Yann LeCun 和 Leon Bottou 在 1990 年代编写的代码。

4.  **本节设置**
(I) 我们导入本节所需的标准库：
```python
import torch
from torch import nn
from d2l import torch as d2l
```

### 7.6.1 LeNet

1.  **LeNet-5 的高层架构**
(I) LeNet-5 由两个主要部分组成：
--- (A) 一个**卷积编码器**（convolutional encoder），由**两个“卷积-池化”块**组成。
--- (B) 一个**稠密块**（dense block），由**三个全连接层**组成。

2.  **核心单元**
(I) LeNet 的基本单元是：一个**卷积层**，后跟一个 **Sigmoid 激活函数**，再后跟一个**平均池化**（`nn.AvgPool2d`）操作。
(II) （d2l 笔记：现在 ReLU 和最大池化（Max-pooling）的效果更好,LeNet当时没有这些知识储备）。

3.  **网络结构与 d2l 实现**
```python
import torch
from torch import nn
from d2l import torch as d2l

# 1. 定义一个辅助函数，用于初始化 CNN 的权重
#    (对应 7.6.1 节)
def init_cnn(module): #@save
    """Initialize weights for CNNs."""
    # 检查模块是全连接层 (Linear) 还是卷积层 (Conv2d)
    if type(module) == nn.Linear or type(module) == nn.Conv2d:
        # 对该层的权重 (module.weight) 使用 Xavier 均匀分布进行初始化
        nn.init.xavier_uniform_(module.weight)

# 2. 定义 LeNet-5 模型
#    继承 d2l.Classifier 以便利用 d2l 的训练和可视化功能
class LeNet(d2l.Classifier): #@save
    """The LeNet-5 model."""    
    # 构造函数, 定义学习率和类别数
    def __init__(self, lr=0.1, num_classes=10):
        # 调用父类 (d2l.Classifier) 的构造函数
        super().__init__()
        # 保存超参数 (lr, num_classes) 以便 later 使用
        self.save_hyperparameters()
        
        # 3. 定义核心网络 (self.net)
        #    使用 nn.Sequential 将所有层按顺序串联
        self.net = nn.Sequential(
            # --- 卷积块 1 ---
            # C1: 6个输出通道, 5x5核, padding=2 (保持 28x28 尺寸)
            nn.LazyConv2d(6, kernel_size=5, padding=2), nn.Sigmoid(),
            # S2: 2x2 平均池化, 步幅=2 (28x28 -> 14x14)
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # --- 卷积块 2 ---
            # C3: 16个输出通道, 5x5核 (14x14 -> 10x10)
            nn.LazyConv2d(16, kernel_size=5), nn.Sigmoid(),
            # S4: 2x2 平均池化, 步幅=2 (10x10 -> 5x5)
            nn.AvgPool2d(kernel_size=2, stride=2),
            
            # --- 全连接块 ---
            # 展平: 将 4D 张量 (batch, 16, 5, 5) 变为 2D 张量 (batch, 16*5*5=400)
            nn.Flatten(),
            # F5: 全连接层, 120个输出单元
            nn.LazyLinear(120), nn.Sigmoid(),
            # F6: 全连接层, 84个输出单元
            nn.LazyLinear(84), nn.Sigmoid(),
            # Output: 输出层, 10个输出单元 (对应 num_classes)
            nn.LazyLinear(num_classes)
        )
        # 注意:
        # 1. nn.LazyConv2d 和 nn.LazyLinear 会自动推断输入通道/维度
        # 2. d2l.Classifier 期待网络保存在 self.net 中
```

4.  **实现说明**
(I) d2l 的实现用 `nn.Sigmoid` 替换了原始 LeNet-5 的高斯激活函数（因为后者很少用）。
(II) `layer_summary` 帮助函数显示了数据在网络中每一步的形状变化，验证了上述尺寸（`28x28` $\rightarrow$ `14x14` $\rightarrow$ `10x10` $\rightarrow$ `5x5`）。
(III) `nn.LazyConv2d` 和 `nn.LazyLinear` 使得我们**不必**手动计算 C3 和 F5 的输入维度（如 16 或 400）。
```python
@d2l.add_to_class(d2l.Classifier) #@save
def layer_summary(self, X_shape):
    X = torch.randn(*X_shape)
    for layer in self.net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)
model = LeNet()
model.layer_summary((1, 1, 28, 28))
'''
Conv2d output shape:    torch.Size([1, 6, 28, 28])
Sigmoid output shape:   torch.Size([1, 6, 28, 28])
AvgPool2d output shape: torch.Size([1, 6, 14, 14])
Conv2d output shape:    torch.Size([1, 16, 10, 10])
Sigmoid output shape:   torch.Size([1, 16, 10, 10])
AvgPool2d output shape: torch.Size([1, 16, 5, 5])
Flatten output shape:   torch.Size([1, 400])
Linear output shape:    torch.Size([1, 120])
Sigmoid output shape:   torch.Size([1, 120])
Linear output shape:    torch.Size([1, 84])
Sigmoid output shape:   torch.Size([1, 84])
Linear output shape:    torch.Size([1, 10])
'''
```

### 7.6.2 Training
具体训练流程如下
```python
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt
# 1. 定义一个辅助函数，用于初始化 CNN 的权重
#    (对应 7.6.1 节)
#def init_cnn(module): #@save
# 2. 定义 LeNet-5 模型
#    继承 d2l.Classifier 以便利用 d2l 的训练和可视化功能
#class LeNet(d2l.Classifier): #@save

if __name__ == '__main__':
    # 设置一个 d2l 训练器, 指定训练 10 个 epochs
    trainer = d2l.Trainer(max_epochs=10, num_gpus=1)

    # 加载 FashionMNIST 数据集, 批量大小为 128
    data = d2l.FashionMNIST(batch_size=128)

    # 创建 LeNet 模型实例, 传入学习率 0.1
    model = LeNet(lr=0.1)

    # --- 关键的初始化步骤 ---
    # model.apply_init 会:
    # 1. 提取一批数据: [next(iter(data.get_dataloader(True)))[0]]
    # 2. 将这批数据“喂”给模型, 以便所有“懒”层 (Lazy layers) 能推断出它们的输入形状
    # 3. 在所有层都被构建后, 调用 init_cnn 函数来初始化权重
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], init_cnn)

    # --- 开始训练 ---
    # 调用 trainer 的 fit 方法, 传入模型和数据
    trainer.fit(model, data)
    plt.show()
```

### 7.6.3 Summary
1.  **本章的进展**
(I) 我们取得了重大进展：从 1980 年代的 MLPs 发展到了 1990 和 2000 年代初的 CNNs。
(II) 像 LeNet-5 这样的架构至今仍然有意义。
(III) LeNet 在 Fashion-MNIST 上的错误率，更接近于 ResNet 这样的高级架构，而不是 MLP。
(IV) 一个主要区别是，更强的计算能力（greater amounts of computation）使得更复杂的架构成为可能。

2.  **实现的简易性**
(I) 第二个区别是**实现的相对容易性**（relative ease）。
(II) 过去需要数月 C++ 和汇编代码的工程挑战，现在在几分钟内就可以完成。
(III) 这种**生产力的巨大提升**（incredible productivity boost）极大地**普及**（democratized）了**深度学习**。
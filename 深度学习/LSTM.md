# LSTM
## 一、回顾RNN
&emsp;&emsp;标准的RNN结构如下：
<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/RNN.png"  style="zoom:30%;" width="100%"/></center>
&emsp;&emsp;我们知道，循环神经网络不过是对序列数据的建模，由大量重复的单元组成，每个单元负责处理一个时间步的 token，一个单元由输入层、隐藏层以及输出层组成，它们两两之间有着直接的前向关系。
&emsp;&emsp;循环神经网络的精髓在于这些单元之间也有着直接的前向关系，且结构也是相同的，所以这种单元间的关联可以画成自回路的形式，如下图：

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/RNN-rolled.png"  style="zoom:30%;" width="30%"/></center>

&emsp;&emsp;因为这种单元间的关联是在隐藏层上实现的，且每个单元隐藏层到输出层的结构较为简单，~~而且为了偷懒~~ 故而我们主要研究输入层到隐藏层的部分，这里的输出也从输出层输出 $\mathbf{o}$​ 变为了隐藏层输出 $\mathbf{h}$​。
&emsp;&emsp;在标准 RNN 中，两个单元间的联系十分简单，即用 tanh 将此时间步的输入与上一个时间步的隐藏层状态连接起来。

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/RNN6.png"  style="zoom:30%;" width="120%"/></center>
&emsp;&emsp;用公式表示则是：
$$
\mathbf{h}^{(t)}=\phi (\mathbf{U}\mathbf{x}^{(t)}+\mathbf{W}\mathbf{h}^{(t-1)}+\mathbf{b})
$$


---

## 二、LSTM的结构
### 1.概述
&emsp;&emsp;LSTM 的整体结构如下：

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM.png"  style="zoom:30%;" width="120%"/></center>

&emsp;&emsp;单个单元的高清大图：（图中均为向量，下同。忘记加粗了...）

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM3.png"  style="zoom:30%;" width="120%"/></center>

&emsp;&emsp;乍一看可能变量很多一下子难以理解，我们先来看各种部件、箭头的含义：
* 箭头：Tensor 前向方向；
* 两个箭头合二为一：两个 Tensor 分别与各自权重点积再加上偏置；
* 箭头一分为二：将值复制；
* $\sigma$ 和 $tanh$：运算层，使流经它们的 Tensor 进行 $sigmoid$ 或 $tanh$ 变换；
* 粉红色圆圈中的 $\times$：与 $sigmoid$​ 共同组成 ***门控***，而后逐点相乘（**Hadamard积**）；
* 粉红色圆圈中的 $+$​：Tensor 加法。

&emsp;&emsp;接下来我们结合上图以及从改进 RNN 的角度来直观感受 LSTM。为了简单考虑，我们将一个时间步下的输入加上各种运算组合而成的系统称为 ***细胞***。
1. **宏观**：单元间的连接关系
    &emsp;&emsp;我们知道，在标准 RNN 中，单元隐藏层间的前向关系是单元后向传播误差的唯一方式，这无疑是危险的，直接导致了 RNN 中严重的梯度消失问题。而从图中可以看到 LSTM 加上了 $C$​ 这根线，拓展了单元间的连接关系。
2. $C$ 线 与 $h$ 线的区别
    &emsp;&emsp;在 RNN 中，我们希望隐藏层的连接能充分提示后文关于前文的信息，于是将每个时间步的详细状态都加入到 $h$ 线中，结果却得不偿失。我们观察上图中 $C$​ 线，只有一个门控与加法，没有 $h$​​ 线复杂的运算，在特殊情况下甚至可能“一通到底”。
3. **微观**：$C$​​​ 线与 $h$​​​ 线互相影响
&emsp;&emsp;在细胞中，$h$​​​ 可以通过两个门控影响 $C_t$​​​，而 $C_t$​​ 可以通过门控决定是否是下个细胞的输入 $\mathbf{h_t}$​​。

### 2.输入：$\mathbf{x_{t}}$​ 与 $\mathbf{h_{t-1}}$​​
&emsp;&emsp;我们从最底层开始。
* $\mathbf{x_t}$​​​：本细胞的输入，形状为 $(1, feature\_size)$​​，而在文本序列中 $x_{t}$​​ 为一个 token，则其 $feature\_size$​​ 即为 Embedding dimension。如下图。（图源：https://zhuanlan.zhihu.com/p/139617364。） 为了简便，记 $\mathbf{x_t}$​ 的形状为 $(1, F)$​​。
* $\mathbf{h_t-1}$​​：上一时刻细胞的隐藏层状态或细胞状态，形状为 $(1,hidden\_size)$​。简记为 $(1,H)$。

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM6.jpg"  style="zoom:30%;" width="120%"/></center>

### 3.门控、门控因子（$f_t$、$i_t$、$o_t$​​​）​与细胞状态 $C_t$
#### 3.1门控与门控因子
&emsp;&emsp;接着我们来看 LSTM 中一个重要概念：**门控**。我们定义：**一个由 $sigmoid$​​ 运算层控制的逐点相乘运算为一个门控运算单元**。如下图：

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM4.png"  style="zoom:30%;" width="25%"/></center>

&emsp;&emsp;门控用公式描述：

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM5.png"  style="zoom:30%;" width="25%"/></center>

$$
\begin{alignat}{2}
\overrightarrow{\gamma}\ {'} = \sigma (\ \overrightarrow{\gamma}\ ) \\
\overrightarrow{\beta}=\overrightarrow{\gamma}\ {'} \odot\overrightarrow{\alpha}
\end{alignat}
$$

&emsp;&emsp;其中 $\odot$​​ 表示哈达玛积，即两个向量或矩阵对应位置元素相乘，部分资料上也记作 $\circ$​​。显然 $\overrightarrow{\alpha},\ \overrightarrow{\beta},\ \overrightarrow{\gamma}$​ 形状应该相同。
&emsp;&emsp;回顾 $sigmoid$​ 函数，它将值压缩在 (0,1) 之间，控制向量 $\overrightarrow{\gamma}$​ 在 $sigmoid$​ 运算后得到门控因子 $\overrightarrow{\gamma}\ {'} $​，它刻画了**原始信息 $\overrightarrow{\alpha}$​ 流过门控的衰减程度**。
&emsp;&emsp;我们回到 LSTM 模型，可以发现其中存在三个门控结构。

##### 3.1.1 遗忘门与遗忘因子 $f_t$​

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM7.png"  style="zoom:30%;" width="25%"/></center>
&emsp;&emsp;我们有：
$$
\begin{alignat}{2}
\boldsymbol{f_{t}}=\sigma(\boldsymbol{{W}_{xf}}&\boldsymbol{x_t}+\boldsymbol{{W}_{hf}}\boldsymbol{h_{t-1}}+\boldsymbol{b_f})\\
\boldsymbol {C_t^{(f)}}&=\boldsymbol{f_{t}} \odot \boldsymbol{C_{t-1}}
\end{alignat}
$$

&emsp;&emsp;其中，$\boldsymbol {C_t^{(f)}}$ 表示前一细胞状态 $\boldsymbol {C_{t-1}}$​ 通过遗忘门后的值。至于“遗忘门”以及接下来的“输入门”、“输出门”是什么意思，我们稍后解释。

##### 3.1.2 输入门与记忆因子 $i_t$​ 
<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM8.png"  style="zoom:30%;" width="25%"/></center>
&emsp;&emsp;有：
$$
\begin{alignat}{2}
\boldsymbol {\tilde{C_t}}&=\sigma(\boldsymbol{{W}_{xC}}\boldsymbol{x_t}+\boldsymbol{{W}_{hC}}\boldsymbol{h_{t-1}}+\boldsymbol{b_C})\\
\boldsymbol{i_{t}}&=\sigma(\boldsymbol{{W}_{xi}}\boldsymbol{x_t}+\boldsymbol{{W}_{hi}}\boldsymbol{h_{t-1}}+\boldsymbol{b_i})\\
& \ \ \ \boldsymbol {C_t^{(i)}}=\boldsymbol {C_t^{(f)}}+\boldsymbol{i_{t}} \odot \boldsymbol{C_{t}^{(f)}}
\end{alignat}
$$
&emsp;&emsp;其中，$\boldsymbol {C_t^{(i)}}$​ 表示前一细胞状态 $\boldsymbol {C_{t-1}}$​​ 通过遗忘门与输入门后的值； $\boldsymbol {\tilde{C_t}}$​​ 称为暂时细胞状态。~~（我起的名称）~~

##### 3.1.2 输出门与输出因子 $o_t$​ 
<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM9.png"  style="zoom:30%;" width="45%"/></center>
&emsp;&emsp;有：
$$
\begin{alignat}{2}
\boldsymbol{o_{t}}=\sigma(\boldsymbol{{W}_{xo}}\boldsymbol{x_t}+&\boldsymbol{{W}_{ho}}\boldsymbol{h_{t-1}}+\boldsymbol{b_o}) \\
\boldsymbol {h_t}=\boldsymbol{o_{t}} &\odot\boldsymbol {C_t^{(i)}}\\
\boldsymbol {C_t}= & \ \boldsymbol {C_t^{(i)}}  \\

\end{alignat}
$$
&emsp;&emsp;其中，$\boldsymbol {C_t}$​​ 称为 $t$​ 时间步下的 ***细胞状态***。

#### 3.2 细胞状态 $C_t$
&emsp;&emsp;接着我们来看 LSTM 中的核心思想：***细胞状态*** 以及它与上述门之间的关联。
&emsp;&emsp;我们在概述中说到，LSTM 相较 RNN 加上了 $C$ 线，拓展了细胞间的关联。我们将所谓 $C$ 线上流动的信息称为细胞状态。观察 $C$ 线在细胞中的流动，它像一条传送带从整个 cell 中穿过，只是做了少量的线性操作。**这种结构能很轻松地实现信息从整个细胞中穿过而不做改变**。

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM10.png"  style="zoom:30%;" width="110%"/></center>


&emsp;&emsp;而这有什么深意呢？显然较少的运算可以让信息流过较远的距离。极端情况下如果中间细胞都没有对 $C$ 进行修改，那么最后的细胞都能原封不动地接受到初始细胞状态。这便是 LSTM 的精髓，$C$ 线相较于RNN中的 $h$，能传播地更远，我们便将细胞状态称为 **长程记忆（Long-Term Memory）**，将原来的隐藏层状态称为 **短程记忆（Short-Term Memory）**，两种信息在细胞中流动，共同组成了 LSTM，这也是它名称的由来：长短期记忆网络（Long Short-Term Memory Networks, LSTM）

### 4.LSTM的前向计算公式
&emsp;&emsp;为了推导出 LSTM是如何前向计算出结果的，我们回到这张整体的结构，捋一捋输入到输出之间经历了什么。

<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM3.png"  style="zoom:30%;" width="65%"/></center>

#### 4.1 遗忘
<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM11.png"  style="zoom:30%;" width="140%"/></center>
&emsp;&emsp;LSTM 的第一步是决定要从上一细胞状态中丢弃什么信息，这通过门控因子 $f_t\ (f\ for\ forget)$实现。思考一个具体的例子，假设我们需要通过 LSTM 构建中译英模型，那么对于谓语形式而言，主语的性别信息尤为重要。当我们又开始描述一个新的主语时，就应适当遗忘旧的主语信息。

#### 4.2 记忆
<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM12.png"  style="zoom:30%;" width="140%"/></center>
&emsp;&emsp;LSTM 的第二步是决定要让多少新的信息加入到细胞状态中。这又包含两个部分：一是加入什么信息？是通过 $tanh$​ 层构建的暂时细胞状态 $\boldsymbol {\tilde{C_t}}$​。而是加多少进细胞状态？这是由门控因子 $i_t$​ 实现。很多资料中都将这个门称为输入门，也是 $i$​ 的名称来源 $i\ for\ input$​​。而我更倾向于称其为记忆门，控制将多少输入的信息记忆进细胞状态中。
&emsp;&emsp;仍以中译英模型为例，我们开始描述一个新的主语时，不仅需要适当遗忘旧的主语信息，还应将新的主语信息加入进细胞状态。

#### 4.3 输出
<center><img src="C:/Users/HJHGJGHHG/Desktop/AI/NLP笔记/深度学习/img/LSTM13.png"  style="zoom:30%;" width="140%"/></center>
&emsp;&emsp;最后，我们需要决定输出了，通过门控因子 $o_t\ (o\ for\ output)$实现控制。
&emsp;&emsp;接着上文中的中译英模型，在翻译到谓语时，细胞状态中包含的主语信息对该细胞的翻译输出影响显著，那么我们需要通过输出门控制相应的输出。

#### 4.4 LSTM的前向计算公式
&emsp;&emsp;我们对所有公式写在一起，就得到了 LSTM 的前向公式了。结合图片与公式，理解应该会深刻很多。
$$
\begin{alignat}{2}
\boldsymbol {forget}:&\\
&\boldsymbol{f_{t}}=\sigma(\boldsymbol{{W}_{xf}}\boldsymbol{x_t}+\boldsymbol{{W}_{hf}}\boldsymbol{h_{t-1}}+\boldsymbol{b_f})\\
&\boldsymbol {C_t^{(f)}}=\boldsymbol{f_{t}} \odot \boldsymbol{C_{t-1}}\\
\boldsymbol {input:}&\\
&\boldsymbol {\tilde{C_t}}=\sigma(\boldsymbol{{W}_{xC}}\boldsymbol{x_t}+\boldsymbol{{W}_{hC}}\boldsymbol{h_{t-1}}+\boldsymbol{b_C})\\
&\boldsymbol{i_{t}}=\sigma(\boldsymbol{{W}_{xi}}\boldsymbol{x_t}+\boldsymbol{{W}_{hi}}\boldsymbol{h_{t-1}}+\boldsymbol{b_i})\\
&\boldsymbol {C_t^{(i)}}=\boldsymbol {C_t^{(f)}}+\boldsymbol{i_{t}} \odot \boldsymbol{C_{t}^{(f)}}\\
\boldsymbol {output:}&\\
&\boldsymbol{o_{t}}=\sigma(\boldsymbol{{W}_{xo}}\boldsymbol{x_t}+\boldsymbol{{W}_{ho}}\boldsymbol{h_{t-1}}+\boldsymbol{b_o}) \\
&\boldsymbol {h_t}=\boldsymbol{o_{t}} \odot\boldsymbol {C_t^{(i)}}\\
&\boldsymbol {C_t}=  \ \boldsymbol {C_t^{(i)}}  \\

\end{alignat}
$$
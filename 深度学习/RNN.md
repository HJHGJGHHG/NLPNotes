# RNN
## 一、我们为什么需要RNN
### 1.从神经网络说起
&emsp;&emsp;从基础的神经网络中知道，神经网络包含输入层、隐层、输出层，通过激活函数控制输出，层与层之间通过权值连接。激活函数是事先确定好的，那么神经网络模型通过训练“学“到的东西就蕴含在“权值“中。我们称**全连接的前馈神经网络**为**线性层**或**全连接层（FC）**，对应 pytorch 中的 nn.Linear。如下图就是一个经典的全连接神经网络：
<center><img src="/img/FC.jpg"  style="zoom:30%;" width="30%"/></center>
&emsp;&emsp;对于表格数值数据，FC的效果不俗。现在我们来看自然语言数据。考虑如下的问题：给定一句话，判断这句话的情感是积极的 (Positive) 还是消极的 (Negative)？显然这是一个情感二分类问题。
&emsp;&emsp;设句子是$S=\{x_0,x_1,\cdots,x_L\}$​，$L$​表示句长，$x_{i}$​表示句中的某个 token，根据粒度不同可以是字或词。一个原始的想法即为：将每个 token 用词向量表示，让其通过一个 Linear(Embedding_dim, 2)，再组合得到一个 $(l, 2)$​ 的 tensor， 最后再通过激活函数层得到形状为 $(1)$​ 的输出，作为给定句子正面或负面的概率结果。结构如下图：

<center><img src="/img/FC2.jpg"  style="zoom:30%;" width="80%"/></center>
&emsp;&emsp;当然这种结构会有严重的问题：
* **第一，当输入语句过长时，参数量巨大**
* **第二也是最严重的问题，忽略了序列关系。**

### 2.序列数据
&emsp;&emsp;序列数据（series data）是在不同时间或空间上收集到的数据，用于所描述现象随时间或空间变化的情况。这类数据反映了某一事物、现象等随时间或空间的变化状态或程度。例如我国GDP从1949到2021每年的数值、股价等就是典型的时序数据。
&emsp;&emsp;而文本数据由于语法与语义的限制，不同token之间不能随意调换顺序，而且后文与前文有一定关系，也可认为是时序数据。然而它与股价、天气等不同的是，每个 token 可认为是特征空间中的离散点。

---

## 二、RNN结构与BPTT （标量版）
### 0.在正式开始之前
&emsp;&emsp;需要说明的是，RNN 作为序列数据的建模方法，不只适用于文本数据。此处限定在文本领域，序列即指文本序列，时间步指某个 token 在序列中的位置。
&emsp;&emsp;另外为了叙述简便，下文公式推导中，假定某一时间步的输入 $x^{(t)}$​ 为标量，相当于省去词向量 Embedding，样本空间即为文本空间。向量版可参考花书，还有[RNN BPTT算法详细推导](https://blog.csdn.net/qq_36033058/article/details/107117030?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522163264221516780261992299%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=163264221516780261992299&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~hot_rank-3-107117030.first_rank_v2_pc_rank_v29&utm_term=BPTT%E7%AE%97%E6%B3%95&spm=1018.2226.3001.4187)以及[RNN与其反向传播算法——BPTT(Backward Propogation Through Time)的详细推导](https://blog.csdn.net/qq_42734797/article/details/111439837)。

### 1.RNN的标准结构
&emsp;&emsp;对于标准的RNN结构，给定序列 $S=\{x^{(0)},x^{(1)},\cdots,x^{(t)}\}$，$S$ 在每一时间步均有一输出 $y_{(i)}$， 则 RNN 结构如下：

<center><img src="/img/RNN.png"  style="zoom:30%;" width="100%"/></center>
&emsp;&emsp;乍一看可能难以理解，我们可以先从左侧的图出发。任何模型都是从样本空间 $\mathcal{X}$​ 到 输出空间 $\mathcal{Y}$​ 的一组映射。对于标准RNN：
* 输入为 $\mathbf{x^{1\times T}} \in \mathcal{X}$​​， 
* 目标输出为 $\mathbf{y^{1\times T}} \in \mathcal{Y}$​​​​​​，
* 隐藏层状态 $\boldsymbol{h^{1\times T}}$，
* 预测输出 $\boldsymbol{o^{1\times T}}$​​，
* 误差 $\boldsymbol{L^{1 \times T}}$，
* 总时间步 $T \in \mathbb{R}$，
* 输入层到隐藏层的权重矩阵 $\mathbf{U} \in \mathbb{R}^{T \times 1}$​​，隐藏层到输出层的权重矩阵 $\mathbf{V} \in \mathbb{R}^{T \times 1}$​​， 及**隐藏层间共享**的权重矩阵 $\mathbf{W} \in \mathbb{R}^{T \times 1}$。​​​​

&emsp;&emsp;箭头代表存在计算关系。如果不看隐藏层上的循环箭头，那与 FC 也没啥区别，而 RNN 的精髓便在于**此时间步的隐藏层状态与上一时间步的隐藏层状态相关，且权值共享。**
&emsp;&emsp;那么将左侧的网络结构**按时间**展开，则有右图的结构。据此我们可以进一步解释：
* 隐藏层随时序相关：随着序列的不断推进，前面的隐层将会影响后面的隐层；
* 权值共享：图中的 W 全是相同的，U 和 V 也一样。虽然 $W$、$U$、$V$ 的形状是 $(T \times 1)$ ，但其中的值都是相同的。

&emsp;&emsp;下面的前向计算公式说明地更加清楚。

### 2.标准RNN的前向计算
&emsp;&emsp;接下来我们给出标准 RNN 结构的前向计算公式。我们都知道神经网络的计算包含**预测**与**训练**，其中**预测**指通过输入 $\mathbf{x}$​​​​ 前向计算出模型的输出结果 $\mathbf{o}$​ 的过程​​；而**训练**指基于预测结果 $\mathbf{o}$​​​ 与目标输出 $\mathbf{y}$​​​ 计算出误差 $\mathbf{L}$​​ 并将其反向传播以调整权重的过程。
&emsp;&emsp;对于上述的 RNN 结构，从时间步 $t=1$​ 到 $t=T$，有如下公式：
$$
\begin{alignat}{2}
h^{(t)}&=\phi (Ux^{(t)}+Wh^{(t-1)}+b) \\
o^{(t)}&=Vh^{(t)}+c \\
\hat y^{(t)}&=\sigma(o^{(t)})
\end{alignat}
$$
&emsp;&emsp;其中： $\phi$​ 为隐藏层激活函数；$\sigma$​ 为输出层激活函数。

### 3.标准RNN的反向传播
&emsp;&emsp;对于 FC，我们是通过**反向传播算法**（Back-Propagation, BP）传递误差并更新权重的。而对于 RNN 结构，只是基于时间反向传播而已（Back-Propagation Through Time, BPTT）。BPTT算法本质还是BP算法，归根结底是**梯度下降法**，那么求各个参数的梯度便成了此算法的核心。
&emsp;&emsp;首先假设：该 RNN 为对文本二分类问题的建模。损失函数为二元交叉熵损失函数，输出层激活函数为sigmoid函数，隐藏层激活函数为tanh函数。也即
$$
\begin{alignat}{2}
L^{(t)}&=-[y^{(t)} \log \hat y^{(t)} +(1-y^{(t)})\log (1-\hat y^{(t)})] \\
L&=\sum^{T}_{t}{L^{(t)}} \\
\hat y^{(t)}&=\sigma(o^{(t)})=sigmoid(o^{(t)}) \\
h^{(t)}&=\phi (Ux^{(t)}+Wh^{(t-1)}+b) \\
&=tanh(Ux^{(t)}+Wh^{(t-1)}+b)
\end{alignat}
$$
&emsp;&emsp;我们需要优化的参数分别是权重U、V、W与偏置c、b。

#### 3.1 V与c
&emsp;&emsp;给定时间步 $t$​，误差 $L^{(t)}$ 是利用预测值 $\hat y^{(t)}$ 与目标值 $y^{(t)}$ 通过损失函数计算而得​。则 $L^{(t)}$ 关于 $V$ 的偏导数：
$$
\frac {\partial L^{(t)}} {\partial V} = \frac {\partial L^{(t)}} {\partial o^{(t)}} \cdot \frac {\partial o^{(t)}} {\partial V}=\frac{\partial L^{(t)}}{\partial \hat y^{(t)}} \cdot \frac{\partial \hat y^{(t)}}{\partial o^{(t)}} \cdot \frac{\partial o^{(t)}}{\partial V}
$$

&emsp;&emsp;有：
$$
\begin{alignat}{2}
\frac{\partial L^{(t)}}{\partial \hat y^{(t)}}&=\frac {-[y^{(t)} \log \hat y^{(t)} +(1-y^{(t)})\log (1-\hat y^{(t)})]} {\partial \hat y^{(t)}}\\
&=-\frac{y^{(t)}}{\hat y^{(t)}} +\frac{1-y^{(t)}}{1-\hat y^{(t)}} \\
&=\frac {\hat y^{(t)}-y^{(t)}}{\hat y^{(t)}(1-\hat y^{(t)})} \\
\frac{\partial \hat y^{(t)}}{\partial o^{(t)}}&=sigmoid^{'}(o^{(t)}) \\
&=sigmoid(o^{(t)})(1-sigmoid(o^{(t)})) \\
&=\hat y^{(t)}(1-\hat y^{(t)}) \\
\frac{\partial o^{(t)}}{\partial V}&=h^{(t)} \\
\\
Hence,\ \ \frac {\partial L} {\partial V}&=\sum^{T}_{t=1}{(\hat y^{(t)}-y^{(t)})h^{(t)}} \\
\frac {\partial L} {\partial c}&=\sum^{T}_{t=1}{(\hat y^{(t)}-y^{(t)})}
\end{alignat}
$$

#### 3.2 W、U与b
&emsp;&emsp;接下来就是 $W, U, b$​ 的梯度计算了，这三者的梯度计算是相对复杂的。注意到 $h^{(i)},i<t$​ 有两个后续节点：$o^{(i)}、h^{(i+1)}$​，所以反向传播时，在某个时刻 $t$​ 的梯度损失由当前位置的输出对应的梯度损失和 $t + 1$​ 时刻的梯度损失两部分共同决定，而 $t + 1$​ 时刻又有类似结论。下图能清楚地说明这一点：

<center><img src="/img/RNN2.png"  style="zoom:30%;" width="80%"/></center>
&emsp;&emsp;首先我们来研究误差对**隐藏层状态**的梯度，亦即 $\frac {\partial L}{\partial h^{(t)}}$​​。我们定义误差记号：
$$
\begin{alignat}{2}
\delta ^{(t)} &= \frac {\partial L}{\partial h^{(t)}} \\
&=\frac {\partial L}{\partial o^{(t)}} \cdot \frac {\partial o^{(t)}}{\partial h^{(t)}}+\frac {\partial L}{\partial h^{(t+1)}} \cdot \frac {\partial h^{(t+1)}}{\partial h^{(t)}}\\
&=V(\hat y^{(t)}-y^{(t)})+\delta ^{(t+1)}\cdot[1-(h^{(t)})^2]\cdot W
\end{alignat}
$$

&emsp;&emsp;现在我们可以很快地写出剩余部分了，时序关系只在 $h^{(i)}$ 上存在， $\frac {\partial h^{(t)}}{\partial U}$​​​等等是不受影响的，也即它们只与本时刻的梯度有关。所以：
$$
\begin{alignat}{2}
\frac {\partial L}{\partial U}&=\sum^{T}_{t=1}{\frac {\partial L^{(t)}}{\partial h^{(t)}}}\cdot {\frac{\partial h^{(t)}}{\partial U}}\\
&=\sum^{T}_{t=1} [1-(h^{(t)})^2]{\delta ^{(t)}} x^{(t)} \\
\frac {\partial L}{\partial W}&={\frac {\partial L^{(t)}}{\partial h^{(t)}}}\cdot {\frac{\partial h^{(t)}}{\partial W}}\\
&=\sum^{T}_{t=1} [1-(h^{(t)})^2]{\delta ^{(t)}} h^{(t-1)}\\
\frac {\partial L}{\partial b}&={\frac {\partial L^{(t)}}{\partial h^{(t)}}}\cdot {\frac{\partial h^{(t)}}{\partial b}}\\
&=\sum^{T}_{t=1} [1-(h^{(t)})^2]{\delta ^{(t)}}
\end{alignat}
$$

## 三、梯度消失与梯度爆炸
### 1.梯度消失（Vanishing Gradient）
&emsp;&emsp;任何深度模型都不可避免地面临**梯度消失**问题。此处不妨以 $\frac {\partial L^{(t)}}{\partial b}$ 为例。由上文（28）式， $t$ 时间步下 $b$ 的更新梯度为 $[1-(h^{(t)})^2]{\delta ^{(t)}}$，展开一层为 $[1-(h^{(t)})^2]\{{V(\hat y^{(t)}-y^{(t)})+\delta ^{(t+1)}\cdot[1-(h^{(t)})^2]\cdot W}\}$，其中两处出现了 $[1-(h^{(t)})^2]$，这显然来源于 $tanh{'}$​。下图给出了 tanh 的函数图像。
<center><img src="/img/tanh.png"  style="zoom:30%;" width="80%"/></center>

&emsp;&emsp;可见其值域为 (-1,1)，导数的值域为 （0,1)，所以 $[1-(h^{(t)})^2]\cdot W$​​ 势必小于1。当网络较深时 $\prod_{k=t}^{T} tanh{'}\cdot W$​​​​ 将趋近于0，也即梯度趋于零。梯度消失就意味消失那一层的参数不再更新，那么那一层就变成了单纯的映射层失去作用了。
&emsp;&emsp;了解这一点后我们再来看一个问题：长程依赖。考虑这样一个句子：“小明今天上班很早，比大明一般早了一个多小时，***他*** 这么早到是为了准备会议”，构建语言模型理解该句子时我们希望知道这个 ***他*** 指代的是谁。而由于 RNN 存在梯度消失的问题，在该时间步可能只有前面几层的更新梯度不趋于0，这导致最初几层的更新梯度极小，参数更新停滞，这样模型就无法捕捉长程依赖关系了。
&emsp;&emsp;需要说明的是，上文以 $tanh$​ 为例说明梯度消失问题，好像最后公式中并没有出现 $sigmoid$ 的导函数形式。它的函数图像为：

<center><img src="/img/sigmoid.png"  style="zoom:30%;" width="80%"/></center>

可以看到它的导数值域更小，而 $\hat y^{(t)}$​​将受其限制，而 $\hat y^{(t)}$​​是存在于更新公式中的。事实上， $sigmoid$ 更易发生梯度消失。

### 2.梯度爆炸（Exploding Gradient）
&emsp;&emsp;相比 RNN 严重的梯度消失问题，梯度爆炸问题并不显著。我们注意到更新梯度中含有 $W,V$ 等参数，当它们初始一个较大的值是时，指数递增将掩盖指数递减的效果，最终导致梯度爆炸。这可以采用梯度裁剪、更好的初始化策略等方法来避免。

### 3.如何缓解梯度消失
* 网络结构方面：RNN会导致梯度消失究其根本，是层间误差仅能通过隐藏层 $h$ 一根线来反向传播。更好的层间连接策略可以有效改善这一点，例如残差连接、LSTM门控等等。
* 激活函数方面：避免使用 $sigmoid$，可以采用 $relu$ 等。

## 四、Pytorch中的RNN模块与实践

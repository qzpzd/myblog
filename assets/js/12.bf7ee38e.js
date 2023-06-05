(window.webpackJsonp=window.webpackJsonp||[]).push([[12],{431:function(t,e,s){"use strict";s.r(e);var a=s(2),_=Object(a.a)({},(function(){var t=this,e=t._self._c;return e("ContentSlotsDistributor",{attrs:{"slot-key":t.$parent.slotKey}},[e("p",[t._v("图像分类算法")]),t._v(" "),e("p",[e("code",[t._v("\x3c!-- more --\x3e")])]),t._v(" "),e("h1",{attrs:{id:"小卷积核应用-vggnet"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#小卷积核应用-vggnet"}},[t._v("#")]),t._v(" 小卷积核应用-VGGNet")]),t._v(" "),e("p",[t._v("利用小卷积核代替大卷积核，感受野不变减少网络的卷积参数量")]),t._v(" "),e("p",[e("strong",[t._v("网络结构")]),t._v("\nVGGNet的网络结构如下图所示。VGGNet包含很多级别的网络，深度从11层到19层不等，比较常用的是VGGNet-16和VGGNet-19。VGGNet把网络分成了5段，每段都把多个3*3的卷积网络串联在一起，每段卷积后面接一个最大池化层，最后面是3个全连接层和一个softmax层。")]),t._v(" "),e("p",[e("img",{attrs:{src:"ico.png",alt:"vggnet"}})]),t._v(" "),e("p",[t._v("原文链接"),e("a",{attrs:{href:"https://blog.csdn.net/u013181595/article/details/80974210",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/u013181595/article/details/80974210"),e("OutboundLink")],1)]),t._v(" "),e("h1",{attrs:{id:"最优局部稀疏结构-inception"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#最优局部稀疏结构-inception"}},[t._v("#")]),t._v(" 最优局部稀疏结构-Inception")]),t._v(" "),e("p",[t._v("以往网络结构通过级联进行堆叠，随着深度的加深容易产生梯度消失，Szegedy提出加深网络的宽度，用1×1、3×3、5×5与最大池化并行方式进行组织，形成一个局部稀疏结构。")]),t._v(" "),e("p",[e("img",{attrs:{src:"https://img-blog.csdnimg.cn/bd6680244f0e4bc493328d7f302f6e75.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_20,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}})]),t._v(" "),e("p",[t._v("inception网络或inception层的作用就是代替人工来确定卷积层中的过滤器类型或者确定是否需要卷积层或者池化层。一个inception模块会将所有的可能叠加起来，这就是inception模块的核心内容。\n"),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/9954364fd5dd46e38269ab56fc7991ce.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_20,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}}),t._v("\n通常称中间的层为瓶颈层，也就是最小网络层。我们先缩小网络，然后在扩大 它。通过两张图的对比可以明显看到，采用了1*1的层的计算成本下降了10倍。那么仅仅大幅缩小表示层规模会不会影响神经网络的性能？事实证明，只要合理构建瓶颈层，那么既可以缩小规模，又不会降低性能，从而大量节省了运算。")]),t._v(" "),e("p",[t._v("原文链接"),e("a",{attrs:{href:"https://blog.csdn.net/u010132497/article/details/80060303",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/u010132497/article/details/80060303"),e("OutboundLink")],1)]),t._v(" "),e("p",[e("strong",[t._v("Inception V1-V4总结：")]),t._v(" "),e("strong",[t._v("Inception V1:")]),t._v("\nInception v1的网络，将1x1，3x3，5x5的conv和3x3的pooling，堆叠在一起，")]),t._v(" "),e("p",[t._v("一方面增加了网络的width，")]),t._v(" "),e("p",[t._v("另一方面增加了网络对尺度的适应性；")]),t._v(" "),e("p",[e("strong",[t._v("Inception V2:")]),t._v("\n一方面了加入了BN层，减少了Internal Covariate Shift（内部neuron的数据分布发生变化），使每一层的输出都规范化到一个N(0, 1)的高斯；\n另外一方面学习VGG用2个3x3的conv替代inception模块中的5x5，既降低了参数数量，也加速计算；")]),t._v(" "),e("p",[e("strong",[t._v("Inception V3:")]),t._v("\nv3一个最重要的改进是分解（Factorization），将7x7分解成两个一维的卷积（1x7,7x1），3x3也是一样（1x3,3x1），")]),t._v(" "),e("p",[t._v("这样的好处，")]),t._v(" "),e("p",[t._v("既可以加速计算（多余的计算能力可以用来加深网络），")]),t._v(" "),e("p",[t._v("又可以将1个conv拆成2个conv，使得网络深度进一步增加，增加了网络的非线性，")]),t._v(" "),e("p",[t._v("还有值得注意的地方是网络输入从224x224变为了299x299，更加精细设计了35x35/17x17/8x8的模块。")]),t._v(" "),e("p",[e("strong",[t._v("Inception V4:")]),t._v("\nv4研究了Inception模块结合Residual Connection能不能有改进？")]),t._v(" "),e("p",[t._v("发现ResNet的结构可以极大地加速训练，同时性能也有提升，得到一个Inception-ResNet v2网络，")]),t._v(" "),e("p",[t._v("同时还设计了一个更深更优化的Inception v4模型，能达到与Inception-ResNet v2相媲美的性能")]),t._v(" "),e("p",[e("strong",[t._v("原文链接")]),t._v("："),e("a",{attrs:{href:"https://blog.csdn.net/sunflower_sara/article/details/80686658",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/sunflower_sara/article/details/80686658"),e("OutboundLink")],1)]),t._v(" "),e("h1",{attrs:{id:"恒等映射残差单元-resnet"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#恒等映射残差单元-resnet"}},[t._v("#")]),t._v(" 恒等映射残差单元-ResNet")]),t._v(" "),e("p",[t._v("ResNet 是在 2015年 由何凯明等几位大神提出，斩获当年ImageNet竞赛中分类任务第一名，目标检测第一名。获得COCO数据集中目标检测第一名，图像分割第一名。")]),t._v(" "),e("p",[e("strong",[t._v("残差单元原理")]),t._v("\nH(x)= F(x)+x\n当网络某一层已经能够提取最佳特征时，后续层试图改变特征x会使得网络的损失变大，为了减少损失使F(x)自动趋于0，此时H(x)= x")]),t._v(" "),e("div",{staticClass:"language-mermaid extra-class"},[e("pre",{pre:!0,attrs:{class:"language-mermaid"}},[e("code",[e("span",{pre:!0,attrs:{class:"token keyword"}},[t._v("graph")]),t._v(" LR\n输入特征 "),e("span",{pre:!0,attrs:{class:"token inter-arrow-label"}},[e("span",{pre:!0,attrs:{class:"token arrow-head arrow operator"}},[t._v("--")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token label property"}},[t._v("x")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token arrow operator"}},[t._v("--\x3e")])]),t._v("1"),e("span",{pre:!0,attrs:{class:"token text string"}},[t._v("(权重层)")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token inter-arrow-label"}},[e("span",{pre:!0,attrs:{class:"token arrow-head arrow operator"}},[t._v("--")]),e("span",{pre:!0,attrs:{class:"token label property"}},[t._v("relu")]),e("span",{pre:!0,attrs:{class:"token arrow operator"}},[t._v("--\x3e")])]),t._v("2"),e("span",{pre:!0,attrs:{class:"token text string"}},[t._v("(权重层)")]),e("span",{pre:!0,attrs:{class:"token inter-arrow-label"}},[e("span",{pre:!0,attrs:{class:"token arrow-head arrow operator"}},[t._v("--")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token label property"}},[t._v("Fx")]),t._v(" "),e("span",{pre:!0,attrs:{class:"token arrow operator"}},[t._v("--\x3e")])]),t._v("+"),e("span",{pre:!0,attrs:{class:"token text string"}},[t._v("(+)")]),e("span",{pre:!0,attrs:{class:"token arrow operator"}},[t._v("--\x3e")]),t._v("H"),e("span",{pre:!0,attrs:{class:"token text string"}},[t._v("[Hx]")]),t._v("\n输入特征"),e("span",{pre:!0,attrs:{class:"token arrow operator"}},[t._v("--\x3e")]),t._v(" +"),e("span",{pre:!0,attrs:{class:"token text string"}},[t._v("(+)")]),t._v("\n\n")])])]),e("p",[e("strong",[t._v("网络结构")]),t._v(" "),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/1031c77f29bb4b168c9751da5a834643.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_20,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}}),t._v(" "),e("strong",[t._v("亮点")])]),t._v(" "),e("p",[t._v("1.超深的网络结构（超过1000层）。\n2.提出residual（残差结构）模块。\n3.使用Batch Normalization 加速训练（丢弃dropout）。")]),t._v(" "),e("p",[e("strong",[t._v("1.采用残差结构的原因")]),t._v("\n1.梯度消失和梯度爆炸\n梯度消失：若每一层的误差梯度小于1，反向传播时，网络越深，梯度越趋近于0\n梯度爆炸：若每一层的误差梯度大于1，反向传播时，网络越深，梯度越来越大")]),t._v(" "),e("p",[e("strong",[t._v("2.退化问题")]),t._v("\n随着层数的增加，预测效果反而越来越差。如下图所示\n"),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/882e60772bd449748fb4032d9ca5d7b0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_20,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}})]),t._v(" "),e("h1",{attrs:{id:"多层密集连接-densenet"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#多层密集连接-densenet"}},[t._v("#")]),t._v(" 多层密集连接-DenseNet")]),t._v(" "),e("p",[t._v("huang等2017年受ResNet启发提出一种更加密集的前馈式跳跃连接，从"),e("strong",[t._v("特征角度")]),t._v("出发，通过"),e("strong",[t._v("增加网络信息流的隐性深层监督")]),t._v("和"),e("strong",[t._v("特征复用")]),t._v("缓解了"),e("strong",[t._v("梯度消失")]),t._v("的问题，同时"),e("strong",[t._v("提升模型的性能")]),t._v("。\n"),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/7b5e1f7cf46c4dfebf83a4555951552d.png",alt:"在这里插入图片描述"}}),t._v("\n在同一个Denseblock中要求feature size保持相同大小,在不同Denseblock之间设置transition layers实现Down sampling, 在作者的实验中transition layer由BN + Conv(1×1) ＋2×2 average-pooling组成")]),t._v(" "),e("p",[e("strong",[t._v("网络结构")]),t._v(" "),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/ec1bc576b8a14d2f88d9309cee1090fe.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_20,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}})]),t._v(" "),e("p",[e("strong",[t._v("DenseNet作为另一种拥有较深层数的卷积神经网络,具有如下优点:")])]),t._v(" "),e("p",[t._v("(1) 相比ResNet拥有更少的参数数量.")]),t._v(" "),e("p",[t._v("(2) 旁路加强了特征的重用.")]),t._v(" "),e("p",[t._v("(3) 网络更易于训练,并具有一定的正则效果.")]),t._v(" "),e("p",[t._v("(4) 缓解了gradient vanishing和model degradation的问题.")]),t._v(" "),e("p",[t._v("论文链接：https：//arxiv.org/pdf/1608.06993.pdf")]),t._v(" "),e("p",[t._v("代码的github链接：https：//github.com/liuzhuang13/DenseNet")]),t._v(" "),e("p",[t._v("MXNet版本代码（有ImageNet预训练模型）：https：  //github.com/miraclewkf/DenseNet")]),t._v(" "),e("table",[e("thead",[e("tr",[e("th",[t._v("原文链接")]),t._v(" "),e("th",[t._v("https://zhuanlan.zhihu.com/p/43057737")])])]),t._v(" "),e("tbody",[e("tr",[e("td",[t._v("原文链接")]),t._v(" "),e("td",[t._v("https://www.jianshu.com/p/8a117f639eef")])])])]),t._v(" "),e("h1",{attrs:{id:"特征通道重标定-senet"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#特征通道重标定-senet"}},[t._v("#")]),t._v(" 特征通道重标定-SENet")]),t._v(" "),e("p",[t._v("SENet是ImageNet 2017（ImageNet收官赛）的冠军模型，和ResNet的出现类似，都在很大程度上减小了之前模型的错误率，并且复杂度低，新增参数和计算量小。")]),t._v(" "),e("p",[t._v("一个可以嵌入到主干网络的子模块，包括"),e("strong",[t._v("压缩、激励、和乘积")]),t._v("，可以学习特征通道之间的关系，将每个特征通道对目标任务的重要性转化为可以学习的参数，根据学习到的参数增强有用的特征通道抑制贡献小的特征通道。\n"),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/75317da823f144cdbe41b696dc3aa837.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_20,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}})]),t._v(" "),e("ol",[e("li",[t._v("Squeeze部分。即为压缩部分，采用一个全局平均池化操作将输入特征图沿着通道维进行压缩，原始feature map的维度为H"),e("em",[t._v("W")]),t._v("C，其中H是高度（Height），W是宽度（width），C是通道数（channel）。Squeeze做的事情是把H"),e("em",[t._v("W")]),t._v("C压缩为1"),e("em",[t._v("1")]),t._v("C，相当于把H"),e("em",[t._v("W压缩成一维了，实际中一般是用global average pooling实现的。H")]),t._v("W压缩成一维后，相当于这一维参数获得了之前H*W全局的视野，感受区域更广。")]),t._v(" "),e("li",[t._v("Excitation部分。得到Squeeze的1"),e("em",[t._v("1")]),t._v("C的表示后，加入一个FC全连接层（Fully Connected），对每个通道的重要性进行预测，得到不同channel的重要性大小后再作用（激励）到之前的feature map的对应channel上，再进行后续操作。")]),t._v(" "),e("li",[t._v("最后是一个 Reweight 的操作，我们将 Excitation 的输出的权重看做是进过特征选择后的每个特征通道的重要性，然后通过乘法逐通道加权到先前的特征上，完成在通道维度上的对原始特征的重标定。")])]),t._v(" "),e("p",[e("strong",[t._v("SENet可以应用到残差结构和密集连接结构中")]),t._v(" "),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/c4fc2fd895c64a6f91b97a6254199b9a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_20,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}})]),t._v(" "),e("p",[t._v("ImageNet分类Top5错误率：")]),t._v(" "),e("p",[t._v("2014 GoogLeNet  6.67%")]),t._v(" "),e("p",[t._v("2015 ResNet      3.57%")]),t._v(" "),e("p",[t._v("2016 ~~~        2.99%")]),t._v(" "),e("p",[t._v("2017 SENet       2.25%")]),t._v(" "),e("table",[e("thead",[e("tr",[e("th",[t._v("SENet官方Caffe实现：")]),t._v(" "),e("th",[t._v("https://github.com/hujie-frank/SENet")])])]),t._v(" "),e("tbody",[e("tr",[e("td",[t._v("PyTorch实现：")]),t._v(" "),e("td",[t._v("https://github.com/moskomule/senet.pytorch")])]),t._v(" "),e("tr",[e("td",[t._v("TensorFlow实现：")]),t._v(" "),e("td",[t._v("https://github.com/taki0112/SENet-Tensorflow")])])])]),t._v(" "),e("p",[t._v("原文链接："),e("a",{attrs:{href:"https://blog.csdn.net/guanxs/article/details/98544872",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/guanxs/article/details/98544872\n"),e("OutboundLink")],1),e("a",{attrs:{href:"https://blog.csdn.net/liuweiyuxiang/article/details/84075343",target:"_blank",rel:"noopener noreferrer"}},[t._v("https://blog.csdn.net/liuweiyuxiang/article/details/84075343"),e("OutboundLink")],1)]),t._v(" "),e("p",[e("strong",[t._v("总结：")]),t._v("\nSE模块将注意力机制引入到深度学习中，且SE本身模块不会对本身结构造成影响，只是多出一条分支，仅增加少量参数，降低了模型的错误率。")]),t._v(" "),e("h1",{attrs:{id:"通道压缩与扩展-squeezenet"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#通道压缩与扩展-squeezenet"}},[t._v("#")]),t._v(" 通道压缩与扩展-SqueezeNet")]),t._v(" "),e("p",[t._v("伯克利和斯坦福2016年提出的轻量级卷积神经网络，代表模型轻量化的开端")]),t._v(" "),e("p",[t._v("主要包括"),e("strong",[t._v("压缩与扩张")])]),t._v(" "),e("p",[e("strong",[t._v("压缩")])]),t._v(" "),e("p",[t._v("利用1×1卷积降维，减少特征图数目，降低模型的参数量")]),t._v(" "),e("p",[e("strong",[t._v("扩张")])]),t._v(" "),e("p",[t._v("利用1×1与3×3卷积进行扩张，还原特征图数量")]),t._v(" "),e("h1",{attrs:{id:"深度可分离卷积-mobilenet"}},[e("a",{staticClass:"header-anchor",attrs:{href:"#深度可分离卷积-mobilenet"}},[t._v("#")]),t._v(" 深度可分离卷积-MobileNet")]),t._v(" "),e("p",[t._v("定位优势：移动设备或者嵌入式设备的轻量级网络，相对同样轻量级的SqueezeNet网络，参数两量近似，性能更好。")]),t._v(" "),e("p",[t._v("主要分为两部分："),e("strong",[t._v("深度通道卷积与逐点卷积")])]),t._v(" "),e("p",[e("strong",[t._v("深度通道卷积")])]),t._v(" "),e("p",[t._v("对于来自上一层的多通道特征图，首先将其全部拆分为单个通道的特征图，分别对他们进行单通道卷积，它只对来自上一层的特征图做了尺寸的调整，而通道数没有发生变化\n"),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/e337dcdb495147a580d7fe8f38ef5a2a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_18,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}})]),t._v(" "),e("p",[e("strong",[t._v("逐点卷积")])]),t._v(" "),e("p",[t._v("因为深度卷积没有融合通道间信息，所以需要配合逐点卷积使用。")]),t._v(" "),e("p",[t._v("采取卷积核1×1大小，滤波器包含了与上一层通道数（"),e("strong",[t._v("即深度卷积通道个数")]),t._v("）一样数量的卷积核。相对于通道而言，对每个通道进行整合卷积（"),e("strong",[t._v("之间是每个通道单独卷积，这里将不同通道利用1×1逐点卷积并进行合并得到特征图")]),t._v("），这又被称之为逐点卷积（Pointwise Convolution）。\n"),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/7a0c21000cc74b0c861d2b66a4edddfc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_18,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}}),t._v(" "),e("strong",[t._v("常规卷积")]),t._v(" "),e("img",{attrs:{src:"https://img-blog.csdnimg.cn/8134f6e9286d48309ea597872dc41805.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBA5Y-k5pyI5ZOl5qygNjY2,size_19,color_FFFFFF,t_70,g_se,x_16",alt:"在这里插入图片描述"}}),t._v(" "),e("strong",[t._v("总结")]),t._v("：")]),t._v(" "),e("p",[t._v("常规卷积与深度可分离卷积不同其实在于深度可分离卷积将卷积过程进行了拆分，把特征图通道与卷积核通道卷积拆分出来。")]),t._v(" "),e("p",[e("strong",[t._v("具有更少的参数量（"),e("a",{attrs:{href:"https://www.cnblogs.com/gshang/p/13548561.html",target:"_blank",rel:"noopener noreferrer"}},[t._v("参数量对比"),e("OutboundLink")],1),t._v("）、计算代价")])]),t._v(" "),e("p",[t._v("参数量计算方式：")]),t._v(" "),e("p",[t._v("P=M1* M2* D* D")]),t._v(" "),e("p",[t._v("注：M1,M2输入输出特征图数量，D卷积核大小")])])}),[],!1,null,null,null);e.default=_.exports}}]);
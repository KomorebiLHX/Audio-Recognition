# 期末项目报告

## 说明

该仓库是大数据案例事务课程的期末项目。项目内容为识别音频是男声还是女声。训练集包含2000个实例，测试集包含500个实例。

## 数据探索

首先，我们需要对数据集进行初步探索，以对数据集有一个初步的认识，为后续的特征工程和模型构建做准备。（探索过程见exploration.ipynb）

### 数据格式

我们使用librosa库读取数据，采用率设为16000，读取后画出波形图

![image-20211124145611579](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124145611579.png)

读取后的信号数组为一维数组，长度根据音频长度改变。

![image-20211124145750777](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124145750777.png)

### 音频时长

我们想要查看各音频的时长分布，使用librosa.get_duration函数。

![image-20211124145850033](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124145850033.png)

画出音频时长的直方图。

<img src="C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124145919508.png" alt="image-20211124145919508" style="zoom:50%;" />

可以看到，大部分音频在2-4s，最长的音频在11s左右。

### 平均基频

根据资料，男生和女生的基频有着显著不同，可以作为一个特征。男声的基音频率大都在100-200HZ之间，而女声则在200-350HZ之间，因此，我们尝试使用pysptk.sptk.swipe函数计算出各个音频的基频。

![image-20211124150235030](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124150235030.png)

我们查看男生平均基频和女生平均基频的均值

<img src="C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124150330318.png" alt="image-20211124150330318" style="zoom:100%;" />

可以看到两者的差异较大，说明平均基频是一个区分度较大的特征。

我们还可以画出男生基频和女生基频的直方图。

女生基频为：

![image-20211124150505834](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124150505834.png)

男生基频为：

![image-20211124150529456](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124150529456.png)

### 梅尔频谱

研究表明，人类对频率的感知并不是线性的，并且对低频信号的感知要比高频信号敏感。梅尔频谱就是在梅尔刻度下的语谱图，是通过语谱图与若干个梅尔滤波器点乘得到的图谱。梅尔刻度是频率的对数刻度：

![img](https://pic1.zhimg.com/80/v2-f43513d0c853e3287a4e1bef5654a7f4_1440w.jpg)

获得梅尔频谱图的具体步骤为：

1. 获取音频的信号数据
2. 对音频信号进行快速傅里叶变换，获得频谱
3. 通过对信号的多个窗口执行快速傅里叶变换得到语谱图
4. 对语谱图将频率刻度转换为梅尔刻度，通过梅尔滤波器变换为梅尔频谱

我们可以用librosa.feature.melspectrogram得到梅尔频谱，我们分别计算出男生的梅尔频谱和女生的梅尔频谱。

![image-20211124152231692](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124152231692.png)

![image-20211124152244950](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124152244950.png)

可以看到，男生的频谱在低频域的分贝较大，女生的频谱在高频域的分贝较大，因此梅尔频谱是一个区分度较大的特征。

### 梅尔倒谱系数

如果我们对梅尔频谱看做是时域信号，先取对数，然后再做一次傅里叶变换（实际上是逆傅里叶变换）， 通过低通滤波器获得频谱包络，那么我们就获得了梅尔倒谱系数MFCC，MFCC就是这帧语音的特征。

![img](https://pic3.zhimg.com/80/v2-c05dd3d7d25a5f29437e5379c68306e6_1440w.jpg)

所谓梅尔倒谱系数指的就是梅尔频谱的包络（Spectral Envelope）， 它可以概括梅尔频谱的变化趋势，作为音频的一大特征。

在实践中，我们使用librosa.feature.mfcc获得MFCC，其中参数n_mfcc表示MFCC的个数。

由于每个音频的时长不同，我们获得的梅尔倒谱系数个数也不同，这样会影响后续的模型架构。因此，为了统一特征的个数，我们在时间方向求平均值，获得n_mfcc个平均梅尔倒谱系数。

同样的，我们计算出男生和女生的梅尔倒谱系数，并且画出系数的直方图。

![image-20211124154032622](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124154032622.png)

![image-20211124154057555](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124154057555.png)

可以看到，男生和女生的梅尔倒谱系数分布存在一定的差异，可以作为音频特征。

## 特征工程

有了上述的数据探索，我们决定将平均基频和梅尔倒谱系数作为数据特征。

为了更好地刻画音频特征我们采用99个平均倒谱系数以及平均基频，一共100个特征作为模型的输入。

我们用pytorch的Dataset类定义数据集，每次从数据集中获取一项，我们都会调用librosa.feature.mfcc函数获取音频的MFCC以及平均基频作为特征。其中，平均基频已经在数据探索环节中计算出来并存储在excel文件中。代码详见dataset.py。

![image-20211124155458609](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124155458609.png)

## 模型架构

我们采用四层神经网络作为模型，这四层的架构为：

+ 输入为100维，输出为200维，激活函数为ReLU
+ 输入为200维，输出为200维，激活函数为ReLU
+ 输入为200维，输出为200维，激活函数为ReLU
+ 输入为200维，输出为2维，激活函数为Sigmoid

最后一层输出的2维结果是每一个类别的分类分数，最后我们会对其进行softmax操作，获得各个类别的概率。

代码详见model.py。

![image-20211124191546265](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124191546265.png)

## 训练过程

训练、评价及测试代码详见main.ipynb，下面作简要描述。

首先，我们将数据集根据9:1的比例分为训练集和测试集（1800个样例作为训练集，200个样例作为测试集），切分过程采用random函数随机分组。

![image-20211124191838045](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124191838045.png)

为了加快训练速度，采用cuda平台进行训练。

![image-20211124192051097](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124192051097.png)

损失函数采用交叉熵损失函数，优化器采用Adam优化器。

![image-20211124192031009](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124192031009.png)

我们设置50个epoch进行训练，每次训练完1个epoch我们都会计算训练集的损失函数、正确率，并且会在测试集上测试一遍模型，计算测试集的准确率，精确率，召回率和F1得分。每当测试集的准确率有所上升，我们会保存一次模型，留到后续以填充真正的无标签测试集。

![image-20211124192416664](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124192416664.png)

训练的具体参数详见config.py。

## 评价指标

我们采用4个评价指标。

### 准确率（accuracy）

指的是预测标签与真实标签相同的比例。

### 精确率（precision）

![[公式]](https://www.zhihu.com/equation?tex=Precision+%5Ctriangleq+%5Cfrac%7BTP%7D%7BTP+%2B+FP%7D)

从预测结果角度出发，描述了二分类器预测出来的正例结果中有多少是真实正例，即该二分类器预测的正例有多少是准确的。

### 召回率（recall）

![[公式]](https://www.zhihu.com/equation?tex=Recall+%5Ctriangleq+%5Cfrac%7BTP%7D%7BTP%2BFN%7D)

从真实结果角度出发，描述了测试集中的真实正例有多少被二分类器挑选了出来，即真实的正例有多少被该二分类器召回。

### F1得分（F1 score）

$$
F1 = \frac{2 (precision\times recall)}{(precision + recall)}
$$

## 模型性能

经过上述训练过程，我们获得了最终的模型。我们对最终模型在测试集上进行了上述4个评价指标的评估。

![image-20211124193454009](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124193454009.png)

模型的准确率为97%，精确率为98.63%，召回率为97.3%，F1得分为97.96%。从混淆矩阵可以看到在200个样例中仅有6个样例分类错误。

## 填充测试集

最后，我们用训练好的模型对测试集进行标签的填充，填充结果详见test.xlsx。

![image-20211124193818379](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124193818379.png)

![image-20211124193850450](C:\Users\LinHengxu\AppData\Roaming\Typora\typora-user-images\image-20211124193850450.png)


# exFM
nexFM -- Flexible(support various feature forms) and high-performance(training and online serving) FM implementation. 

默认15维，如果指定其他维度，用make DIM=xxx进行编译。

1. make DIM=31
FM 隐向量V的dim，可以用变长数组实现：
1. vector<double>
2. 变长数组：struct {double w, doubleV[] }， 好处是节省指针，一次性开辟一篇连续内存，缺点是代码可读性差，对于sgdm、adam、ftrl之类的优化器，因为需要位置保存
3. 宏定义：好处，代码可读性好，g++ -O3优化选项可以进行循环展开，因为对V的遍历操作非常多，所以对V的循环进行循环展开能显著提升效率。（for循环不利于cpu流水线运作，依赖于分支预测）



XXXSolver:  public SolverInterface

paramUnit

paramContainerInterface
paramContainer<ParamUnitType> : public paramContainerInterface



sgdm

adagrad：

adam：
adamw



ftrl

容易适配特征，不用做什么特征工程  label1, label2,  x1:v1,   x2:v21|v22|v23|v24,   x3:v3 ， 这种就行


FTRL里面的n z是什么意思，在论文里是什么单词，怎么描述的，然后把代码里面的变量名改一下

精度方面，优化学习器
attention：共享embedding，可以指定N个行为序列，每个行为序列与对应的targetField进行attention。
多目标 


batch_size：
现我们在每一次epoch迭代的时候，都会打乱数据，随机分割数据集。
这是因为神经网络参数多，学习能力强，如果不乱序的话，同一个组合的batch反复出现，模型有可能会“记住”这些样本的次序，从而影响泛化能力。
要做shuf和batch


cpu向量指令 性能优化

adam需要batchsize，其次是sgdm，FTRL不需要：


Batch Size=1，梯度变来变去，非常不准确，网络很难收敛，需要较小的学习率以保持稳定性。batch_size为1时，adam学习率0.001很难学好，这是为什么学习率要低两个数量级的原因。(官方推荐lr=0.001，但是不支持batch_size的时候，lr要调到1e-5以下)

3、Batch Size增大，梯度变准确，
4、Batch Size增大，梯度已经非常准确，再增加Batch Size也没有用
注意：Batch Size增大了，要到达相同的准确度，必须要增大epoch。
https://blog.csdn.net/qq_34886403/article/details/82558399


支持batch_size之后，lr太大还是学不好：
# sgdm lr=0.01就完全学不动，即使是batch_size=1024，如果lr=0.1，auc一直0.5。 
# adam 也是 lr=0.0001比较好，0.01和0.001学不出来




parameter synchronize：
Mutex_t通过宏定义控制：
    #ifdef _PREDICT_VER_
    typedef NullMutex Mutex_t;
    #else
    typedef PthreadMutex Mutex_t;
    #endif



shuffle:
exFM的shuffle只是小范围shufle，具体的讲是给多个worker线程分发的时候不是采用轮训分发而是随机分发，只能避免多个epoch内每个batch组合不变的情况。
在使用自适应学习率算法(adagrad, adam)的时候，可以考虑对训练集进行整体的打散，可以避免某些特征集中出现，而导致的有时学习过度、有时学习不足，使得下降方向出现偏差的问题




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


adam


ftrl


FTRL里面的n z是什么意思，在论文里是什么单词，怎么描述的，然后把代码里面的变量名改一下

共享embedding

精度方面，优化学习器

多目标（工业级很实用，是一个有吸引力的点。另一个吸引的点是容易适配特征，不用做什么特征工程  label1, label2,  x1:v1,   x2:v21|v22|v23|v24,   x3:v3 ， 这种就行）



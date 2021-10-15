# exFM - FM with some useful extensions

## 产出特征配置

根据你的训练数据，产出一个简单的特征处理的配置文件，该配置文件将定义好你所需要的每个连续特征、离散特征、序列特征的处理方式。

参照示例conf_csv.py和conf_criteo.py。以下是配置内容的解释。你可以配置好第1和2项，后续的都可以采用默认的配置，依次来生成一个默认的模型训练特征配置。

1. 配置训练数据的格式。

   1. 如果你的训练数据是csv：
      1. 配置data_formart =  "csv"
      2. 通过feat_sep, feat_values_sep列之间的分隔符、序列特征中多个值的分隔符
      3. 确保数据第一行为表头（各列列名），后续配置中配置的特征名称将与csv的表头对应。
      4. 示例：conf_criteo.py
   2. 如果你的训练数据是libsvm格式：
      1. 配置data_formart =  "libsvm"
      2. 通过feat_sep, feat_kv_sep, feat_values_sep配置特征之间的分隔符、特征key和value之间的分隔符、varlen_sparse_feature中多个值的分隔符。

2. 配置你所需要的特征。通过以下5个list配置你所需要的5种特征：

   | 配置项                      | 特征类型           |
   | --------------------------- | ------------------ |
   | dense_feat_list             | 连续特征           |
   | sparse_id_feat_list         | 数值型离散特征     |
   | varlen_sparse_id_feat_list  | 数值型序列特征     |
   | sparse_str_feat_list        | 字符串类型离散特征 |
   | varlen_sparse_str_feat_list | 字符串类型序列特征 |

3. 连续特征相关配置：

   1. 通过dense_feat_wide_splits和dense_feat_freq_splits配置等宽分桶和等频分桶的分桶数。

4. 离散特征相关配置（同样适用于序列特征）：

   1. 通过default_id_of_sparse_feat，unknown_id_of_sparse_feat配置特征的缺省ID和未知ID。
   2. 通过sparse_feat_mapping_type配置离散特征的ID映射方式，支持dict/hash/orig_id。

5. 此时你可以运行make_feat_config.py，来生成一个特征处理配置文件：

   ```
   cat ../data/train.csv | python3 make_feat_config.py -o criteo --cpu_num 6
   ```

6. 这是将产生一套简单的特征处理配置文件，特征配置文件的规范和解释见文档[特征处理配置文件](https://github.com/tangwang/exFM/blob/main/docs/feature_config.md)。这时可以直接到下一步训练模型，也可以对特征处理配置文件进行一些调整，比如：

   1. 对于连续特征，为不同的特征指定不同的分桶数。
   2. 对于离散特征或序列特征，调整映射方式（支持 dict / hash / orig_id ）
   3. 对于序列特征，调整max_len。

## 训练模型

1. 编译train程序：make即可。FM的embedding维数不支持参数配置，而是在编译时指定，默认为DIM=15，如果要修改维数，比如改为32维，使用make DIM=32。

2. 配置config/train.conf。根据config/train.conf中的注释进行配置即可，主要要配置的内容有：

   1. 训练数据格式：训练数据格式的配置与上面 [产出特征配置]() 中的配置方法一样。

   2. 配置训练数据地址。如果不配置，则程序会读取标准输入。

   3. 配置validation数据地址（可选）

   4. 特征处理配置配置文件路径：配上make_feat_config.py输出的文件夹名称就行，比如 feat_cfg = criteo （criteo是在config目录下用make_feat_config.py产出的特征配置）

   5. 配置优化器： 比如 solver  = adam （目前支持adam / adagrad / rmpprop / ftrl / sgdm )各优化器的超参数都可以在该配置文件进行调整。

   6. 启动程序进行训练。以上所有的配置，都可以放在命令行中，命令行中的参数将覆盖配置文件中的参数。比如：

      ```
      bin/train solver=adam adam.lr=0.001 batch_size=800 feat_conf=criteo threads=30 train=../data/train.csv valid=../data/valid.csv om=model
      ```

   7. 





公开数据集：
https://github.com/ycjuan/kaggle-2014-criteo
https://github.com/ycjuan/libffm



默认15维，如果指定其他维度，用make DIM=xxx进行编译。



XXXSolver:  public SolverInterface

paramUnit

paramContainerInterface
paramContainer<ParamUnitType> : public paramContainerInterface


sgdm

adagrad：

adam：
adamw



ftrl
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
sgdm lr=0.01就完全学不动，即使是batch_size=1024，如果lr=0.1，auc一直0.5。 
adam 也是 lr=0.0001比较好，0.01和0.001学不出来


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



cityhash 的优化选项：
./configure
make all check CXXFLAGS="-g -O3"
sudo make install

Or, if your system has the CRC32 instruction, and you want to build everything:

./configure --enable-sse4.2
make all check CXXFLAGS="-g -O3 -msse4.2"
sudo make install

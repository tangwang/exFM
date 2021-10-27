# exFM - FM with some useful extensions

## 产出特征配置

特征方面支持连续特征（denseFeat）、稀疏特征（sparseFeat 支持ID型和字符串类型）、变长特征（varlenSparseFeat），内置了一套工业界比较标准、实用的特征处理的方法，你可以用以下工具产出一个特征处理的配置文件，该配置文件将定义好你所需要的每个连续特征、离散特征、序列特征的处理方式。

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

6. 这时将产生一套简单的特征处理配置文件，特征配置文件的规范和解释见文档[特征处理配置文件](https://github.com/tangwang/exFM/blob/main/docs/feature_config.md)。这时可以直接到下一步训练模型，也可以对特征处理配置文件进行一些调整，比如：

   1. 对于连续特征，为不同的特征指定不同的分桶数。
   2. 对于离散特征或序列特征，调整映射方式（支持 dict / hash / orig_id ）
   3. 对于序列特征，调整max_len。

## 训练模型

1. 编译train程序：make即可。FM的embedding维数不支持参数配置，而是在编译时指定，默认为DIM=15，如果要修改维数，比如改为32维，使用make DIM=32。

2. 配置config/train.conf。根据config/train.conf中的注释进行配置即可，主要要配置的内容有：

   1. 训练数据格式：data_formart 和csv/libsvm相关格式配置的配置方法与上面的一致。

   2. 配置训练数据地址。如果不配置，则程序会读取标准输入。

   3. 配置validation数据地址（可选）

   4. 特征处理配置配置文件路径：配上make_feat_config.py输出的文件夹名称就行，比如 feat_cfg = criteo （criteo是在config目录下用make_feat_config.py产出的特征配置）

   5. 配置优化器： 比如 solver  = adam （目前支持adam / adagrad / rmpprop / ftrl / sgdm )各优化器的超参数都可以在该配置文件进行调整。

      建议使用ftrl/adam优化器。
      FM模型的参数较适合于用FTRL进行更新，笔者试了几个数据集都在ftrl上取得最优，adam和adagrad的效果通常非常接近于FTRL，或者持平，少数数据集上adam/adagrad取得最优。
      但是FTRL不需要batchsize，所以batchSize不能过高（设为1或者10以下）。adam和adagrad的batchSize不能过小（否则需要极低的学习率，较难调参），可以设置为500~2000。所以选择adam/adagrad的话参数更新频次更低，训练速度比使用FTRL快很多。
      sgdm没有精心优化和调试，试了一些数据集都比FTRL/adam/adagrad效果有明显差距。

   6. 配置threads参数指定训练的线程数。

   7. 启动程序进行训练。以上所有的配置，都可以放在命令行中，命令行中的参数将覆盖配置文件中的参数。比如：
   
      ```
      bin/train solver=adam adam.lr=0.001 batch_size=800 feat_conf=criteo threads=30 train=../data/train.csv valid=../data/valid.csv om=model
      ```

   8. 程序输出：模型输出路径通过om指定，支持按文本和二进制格式输出（通过mf=txt / om=bin指定）。

      如果sparse特征指定的映射方式为dynamc_dict，则会将特征ID映射词典一并输出。
   
      

## examples

1. criteo

   ```
   git clone xxx
   cd config
   # 准备你的训练数据
   # 根据你的训练数据配置conf.py
   # 产出特征处理配置
   cat ../data/train.csv | python3 make_feat_config.py -o criteo --cpu_num 6
   cd ..
   
   # 根据你的训练数据配置config/train.conf
   # 启动训练
   bin/train solver=adam batch_size=800 feat_conf=criteo threads=30 train=../data/train.csv valid=../data/valid.csv om=model
   ```

---



公开数据集：
https://github.com/ycjuan/kaggle-2014-criteo
https://github.com/ycjuan/libffm

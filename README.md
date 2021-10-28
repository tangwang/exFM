# exFM - FM with some useful extensions

## features

支持对连续型特征的离散化，包括等频和等宽分桶，也可以基于对特征的理解配置非线性映射（取对数、指数）后进行等宽分桶。

支持对离散特征自动的进行ID映射，映射方式支持dict（事先统计好映射词典）、dynamic_dict（边训练变更新映射词典）、hash、orig_id（直接使用原始值）。原始值支持数值、字符串。

支持序列特征，对序列特征的处理方式为sum_pooling/avg_pooling。

基于特征类型的配置，和输入的数据，通过make_feat_config.py自动产生一个用于训练和线上预估的特征处理配置文件。

支持ftrl和流式训练，支持adam/adagrad等主流优化器和batch训练。

高性能：使用cpu训练的情况下，性能相比TensorFlow和pytorch的FM、difacto、alphaFM都要高很多。

支持模型初始化和模型dump。

支持多个特征共享embedding：itemID、tagID通常会出现在多个field（出现在item侧、user侧），如果ID特别稀疏配置为共享参数会有好处，但是该特性需要配合deep网络结构使用，见我的另一个项目[deepFM](https://github.com/tangwang/deepFM)。

### examples

这里给出一个极简的样例。详细说明见下面的usage。

```
git clone https://github.com/tangwang/exFM.git
# 编译，通过dim配置embedding_size，生成bin/train
make dim=15 -j 4

# 准备你的训练数据
cd data
xz -d criteo_sampled_data.csv.tar.xz
tar -xf criteo_sampled_data.csv.tar
cd -

# 利用make_conf.py和训练数据，生成一个特征转换配置文件。 也可以基于你对你的特征的理解自行编写一个特征处理的配置文件。
cd config
cp conf_criteo.py conf.py
cat ../data/criteo_sampled_data.csv.train ../data/criteo_sampled_data.csv.test  | python3 make_feat_conf.py -o criteo --cpu_num 4
cd -

# 根据你的训练数据配置config/train.conf
# 启动训练
bin/train data_formart=csv feat_sep=, feat_cfg=criteo train=data/criteo_sampled_data.csv.train valid=data/criteo_sampled_data.csv.test threads=4 verbose=0 epoch=20 solver=adam batch_size=1000 mf=txt om=model6
# dim=15时，test AUC 0.7764，在我的4核（至强E-2224G CPU）机器上训练速度为26万样本/S。

bin/train data_formart=csv feat_sep=, feat_cfg=criteo train=data/criteo_sampled_data.csv.train valid=data/criteo_sampled_data.csv.test threads=4 verbose=0 epoch=30 solver=ftrl batch_size=10
# 使用FTRL batch_size=10，test AUC 0.7783
```

## usage

### 产出特征配置

特征方面支持连续特征（denseFeat）、稀疏特征（sparseFeat 支持ID型和字符串类型）、变长特征（varlenSparseFeat），内置了一套工业界比较标准、实用的特征处理的方法，你可以用以下工具产出一个特征处理的配置文件，该配置文件将定义好你所需要的每个连续特征、离散特征、序列特征的处理方式。

示例：

如果你的数据是这样

```
label,item_id,chanel,item_tags,item_clicks,item_price,user_click_list,user_age
0,123,aaa,信托|记账|酒店,2521,0.3,342|5212|839,24
1,3423,bcd,培训|租房,342,1.2,44|3422|34|8,33
0,332,,,342,1.2,14|3343|452|36|8,33
```

可以这样配置（将以下内容写入config/conf.py）：

```
# 训练数据格式，支持csv和libsvm，如果是csv格式必须确保第一行为表头（各列列名）,并且第一列为label(0/1, 或者-1/1)。
data_formart =  "csv"

# 对于csv和libsvm都需要配置域分隔符和序列特征中多个值的分隔符
feat_sep = ','
feat_values_sep = '|'

dense_feat_list = ['item_clicks', 'item_price', 'user_age'] # 配置你的连续特征
sparse_id_feat_list = ['item_id']                           # 数值型离散特征
sparse_str_feat_list = ['chanel']                           # 字符串类型离散特征
varlen_sparse_id_feat_list = ['user_click_list']            # 数值型序列特征
varlen_sparse_str_feat_list = ['item_tags']                 # 字符串类型序列特征

# 对于连续特征，配置等宽分桶和等频分桶的分桶数，以如下配置为例，则一个连续型特征会按等宽分桶和等频分桶分别离散化为2个ID特征，共4个ID特征
dense_feat_wide_splits = [10, 25]
dense_feat_freq_splits = [10, 25]

default_value_of_dense_feat = 0

min_freq_for_sparse_feat_dict = 0 # 特征出现次数大于该值时，才加入特征ID映射词典

seq_feat_max_len = 30
seq_feat_pooling_type = "sum" # 暂时只支持sum和avg

# 配置离散特征的ID映射方式，支持dict/dynamic_dict/hash/orig_id
sparse_feat_mapping_type = "dict"
```

 如果数据是libsvm格式，只需要将data_formart配置为"libsvm"，参考[conf_libsvm.py](https://github.com/tangwang/deepFM/blob/main/config/conf_libsvm.py)



1. 此时你可以运行make_feat_config.py，来生成一个特征处理配置文件：

   ```
   cd config
   # 按上面的方式配置conf.py中的内容
   cat ../data/train.csv | python3 make_feat_conf.py -o criteo --cpu_num 6
   # 此时将生成目录criteo，里面有一个训练使用的特征配置文件，和离散特征的ID映射词典。 训练的使用通过 feat_cfg=criteo 来使用该特征配置。
   ```

6. 这时将在criteo目录下产生一套简单的特征处理配置文件，特征配置文件的规范和解释见文档[特征处理配置文件](https://github.com/tangwang/exFM/blob/main/docs/feature_config.md)。这时可以直接到下一步训练模型，也可以对特征处理配置文件进行一些调整，比如：

   1. 对于连续特征，为不同的特征指定不同的离散化分桶数。
   2. 对于连续特征，根据各自数值分布做不同的处理，比如对于点击频次做log处理然后等宽分桶，对item的上架天数做0.5次方然后做等宽分桶。
   3. 对于离散特征或序列特征，调整映射方式（支持 dict / dynamic_dict / hash / orig_id ）。
   4. 对于序列特征，调整长度限定max_len。

### 训练模型

1. 编译train程序：make即可。FM的embedding维数不支持参数配置，而是在编译时指定，默认为DIM=15，如果要修改维数，比如改为32维，使用make DIM=32。

2. 配置config/train.conf。根据config/train.conf中的注释进行配置即可，主要要配置的内容有：

   1. 训练数据格式：data_formart 和csv/libsvm相关格式配置的配置方法与上面的一致。

   2. 配置训练数据地址。如果不配置，则程序会读取标准输入。

   3. 配置validation数据地址（可选）

   4. 特征处理配置配置文件路径：配上make_feat_config.py输出的文件夹名称就行，比如 feat_cfg = criteo （criteo是在config目录下用make_feat_config.py产出的特征配置）

   5. 配置优化器： 比如 solver  = adam （目前支持adam / adagrad / rmpprop / ftrl / sgdm )各优化器的超参数都可以在该配置文件进行调整。

      建议使用ftrl/adam优化器。
      FM模型适用于大规模稀疏特征，参数较适合于用FTRL进行更新，所以通常能在ftrl上取得最优效果，但是基于adam和adagrad+mini_batch也能基本上与FTRL的效果持平。因为FTRL不需要batch_size，batch_size需要设置为1或者10以下，adam和adagrad的batch_size不能过小（否则需要极低的学习率，较难调参），可以设置为500~2000，所以采用adam/adagrad的话参数更新频次更低，训练速度比FTRL快一些。
      sgdm没有精心优化和调试，试了一些数据集都比FTRL/adam/adagrad效果有明显差距。
      
   6. 配置threads参数指定训练的线程数。
   
   7. 启动程序进行训练。以上所有的配置，都可以放在命令行中，命令行中的参数将覆盖配置文件中的参数。比如：
   
      ```
      bin/train solver=adam adam.lr=0.001 batch_size=800 feat_conf=criteo threads=30 train=../data/train.csv valid=../data/valid.csv om=model_20211028
      ```
   
   8. 程序输出：模型输出路径通过om指定，支持按文本和二进制格式输出（通过mf=txt / om=bin指定）。
   
      如果sparse特征指定的映射方式为dynamc_dict，则会将特征ID映射词典一并输出。
   


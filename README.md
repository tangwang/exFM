# exFM - FM model with some useful extensions

## features

易用：通常拿到一份数据，可能要做很多特征工程，比如离散的特征有ID型的、字符串的、变长（多值/序列）的，都需要做些特征处理，连续特征可能要做离散化或者非线性映射等，然后要基于TensorFlow/pytorch等框架编写一些训练的代码。该项目就是通过定义一套特征处理配置文件规范，提供一个配置产出工具，一套FM训练和预测工具，从而使得这些工作基本可用省去。

可靠：针对FM对内置的adamW/FTRL等优化器和batch训练进行了精心的调参，使用默认配置可以得到一个不错的baseline。已经在千万日活的业务体中用于召回和粗排。

高效：使用cpu的情况下，训练速度相比TensorFlow/pytorch实现的FM快很多倍。

### examples

这里给出一个极简的样例。详细说明见下面的usage。

```
git clone https://github.com/tangwang/exFM.git --depth=1
# 编译，通过dim配置embedding_size，生成bin/train
cd exFM
make dim=15 -j 4

# 准备你的训练数据，这里使用项目附带的一个60万criteo数据集
cd data
xz -d criteo_sampled_data.csv.tar.xz
tar -xf criteo_sampled_data.csv.tar
cd -

# 利用make_conf.py和训练数据，生成一个特征转换配置文件。 也可以基于你对你的特征的理解自行编写一个特征处理的配置文件。
cpu_num=`cat /proc/cpuinfo | grep processor | wc -l`
cd config
cp conf_criteo.py conf.py
# 可以使用部分数据，比如top10w行来产出一个特征转换的配置
cat ../data/criteo_sampled_data.csv.train ../data/criteo_sampled_data.csv.test  | python3 make_feat_conf.py -o criteo --cpu_num $cpu_num
cd -

# 打印帮助
bin/train h

# train （配置文件为config/train.conf，可以通过命令行参数补充或覆盖配置文件中的配置项）
bin/train data_formart=csv feat_sep=, feat_cfg=criteo train=data/criteo_sampled_data.csv.train valid=data/criteo_sampled_data.csv.test threads=$cpu_num verbose=1 epoch=20 solver=adam batch_size=1000 mf=txt om=model_1029_txt
# dim=15时，test AUC 0.7765，在我的4核（至强E-2224G CPU）机器上训练速度为25万样本/S。

bin/train data_formart=csv feat_sep=, feat_cfg=criteo train=data/criteo_sampled_data.csv.train valid=data/criteo_sampled_data.csv.test threads=$cpu_num verbose=1 epoch=30 solver=ftrl batch_size=10
# 使用FTRL batch_size=10，test AUC 0.7783

# predict （也会config/train.conf）
# 因为criteo_sampled_data.csv.test没有header line，所以需要通过csv_columns配置列名称
cat data/criteo_sampled_data.csv.test | bin/predict data_formart=csv feat_sep=, feat_cfg=criteo  verbose=0 mf=txt im=model_1029_txt verbose=0 csv_columns=label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26
```

## usage

### 产出特征配置

你只需要配置连续特征、稀疏特征、变长特征（序列特征）的特征名称，然后基于make_feat_conf.py工具自动产出一个特征处理的配置文件，该配置文件将定义好你所需要的每个特征的处理方式。

make_feat_conf.py的使用方式：

1. 准备你的训练数据，这里以csv格式的数据为例：

```
label,item_id,chanel,item_tags,item_clicks,item_price,user_click_list,user_age
0,123,aaa,信托|记账|酒店,2521,0.3,342|5212|839,24
1,3423,bcd,培训|租房,342,1.2,44|3422|34|8,33
0,332,,,342,1.2,14|3343|452|36|8,33
```

2. 可以这样配置你的config/conf.py：

```
# 训练数据格式，支持csv和libsvm，如果是csv格式必须确保第一行为表头（各列列名）,并且第一列为label
# label为0/1  或者-1/1，特征取值支持连续特征(float), 离散特征(int / string), 序列特征(list of int / string)
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

seq_feat_max_len = 30         # 序列特征最大长度
seq_feat_pooling_type = "sum" # 暂时只支持sum和avg

# 配置离散特征的ID映射方式，支持dict/dynamic_dict/hash/orig_id
sparse_feat_mapping_type = "dict"
```

​	 如果数据是libsvm格式，将data_formart配置为"libsvm"，参考[conf_libsvm.py](https://github.com/tangwang/deepFM/blob/main/config/conf_libsvm.py)

3. 运行make_feat_conf.py来生成一个特征处理配置文件：

```
cd config
# 按上面的方式配置conf.py中的内容
cat ../data/train.csv | python3 make_feat_conf.py -o simple_feat_conf --cpu_num 6
# 此时将生成目录simple_feat_conf，里面有一个训练使用的特征配置文件，和离散特征的ID映射词典。 训练的使用通过 feat_cfg=simple_feat_conf 来使用该特征配置。
```

这时将在simple_feat_conf目录下产生一套简单的特征处理配置文件，特征配置文件的规范和解释见文档[特征处理配置文件](https://github.com/tangwang/exFM/blob/main/docs/feature_config.md)。这时可以直接到下一步训练模型，也可以对特征处理配置文件进行一些调整，比如：
1. 对于连续特征，都自动配上了2种等宽分桶+2种等频分桶的离散化方式，你可以对某些特征做细化的调整，或者根据各自的取值分布做不同的处理，典型的，比如对于点击频次做log处理然后等宽分桶，对item的上架天数做0.5次方然后做等宽分桶，等。
2. 对于离散特征或序列特征，调整映射方式（支持 dict / dynamic_dict / hash / orig_id ）。
3. 对于序列特征，调整长度限定max_len。

借助make_feat_conf.py工具来产生特征处理配置文件的好处是，它自动的帮你跑出了每个特征的分布，并产出了一套基本可用的特征处理配置。你也可以基于[特征处理配置文件](https://github.com/tangwang/exFM/blob/main/docs/feature_config.md)直接编写一个特征处理的配置文件。

### train

1. 编译train程序：make即可。FM的embedding维数不支持参数配置，而是在编译时指定，默认为dim=15，如果要修改维数，比如改为32维，使用make dim=32。

2. 配置config/train.conf。根据config/train.conf中的注释进行配置即可，主要要配置的内容有：

   1. 训练数据格式：data_formart 和csv/libsvm相关格式配置的配置方法与上面的一致。

   2. 配置训练数据地址。如果不配置，则程序会读取标准输入。

   3. 配置validation数据地址（可选）

   4. 特征处理配置配置文件路径：配上make_feat_conf.py输出的文件夹名称就行，比如 feat_cfg=simple_feat_conf

   5. 配置优化器： 比如 solver  = adam （目前支持adam (adamW) / adagrad / rmpprop / ftrl / sgdm )。

      建议使用ftrl/adam。
      FM模型的参数通常是大规模稀疏特征的参数，较适合于用FTRL进行更新，所以通常能在ftrl上取得最优效果。FTRL适合逐个样本的流式训练，所以batch_size不能过大，可设为1或者10左右。
      
      adam / adamW / adagrad  / rmsprop + mini_batch也能基本上与FTRL的效果持平，batch_size不能过小（否则需要极低的学习率，较难调参），可以设置为1000左右。
      sgdm没有仔细调试，试了一些数据集效果较差。
      
      各优化器的超参数都可以在配置文件中修改，通常情况下采用默认的即可。
      
   6. 配置threads参数指定训练的线程数。
   
   7. 启动程序进行训练。以上所有的配置，都可以放在命令行中，命令行中的参数将覆盖配置文件中的参数。比如：
   
      ```
      bin/train solver=adam adam.lr=0.001 batch_size=800 feat_conf=simple_feat_conf threads=30 train=../data/train.csv valid=../data/valid.csv om=model_20211028
      ```
   
   8. 程序输出：模型输出路径通过om指定，支持按文本和二进制格式输出（通过mf=txt 或者 mf=bin指定）。
   
      如果sparse特征指定的映射方式为dynamc_dict，则会将特征ID映射词典一并输出。

### predict

1. #### 离线批量预估

   依赖：

   ​	特征配置 feat_cfg=xxx

   ​	加载模型 im=xxx

   ```
   # 因为项目中附带的criteo_sampled_data.csv.test没有header line，补充一下
   head -1 data/criteo_sampled_data.csv.train > data/for_predict.csv
   cat data/criteo_sampled_data.csv.test >> data/for_predict.csv
   cat data/for_predict.csv | bin/predict data_formart=csv feat_sep=, feat_cfg=criteo  verbose=0 mf=txt im=model_1029_txt verbose=0 
   ```
   
2. #### 在线预估

   1. ##### 动态库(lib/fm_pred.so) 

      **该部分功能未使用过、未测试，使用前请做测试，特别是通过java调用的话需要做长时间压测，有问题可以联系作者。**

      编译后会生成一个lib目录，包括一个so动态库 + include文件。

      ###### c++项目的调用方法：

      1）创建FmModel对象（多线程/多进程共享一个即可）：

      ```c++
      #include "lib_fm_pred.h"
      
      // config/train.conf必须配置的参数有：
      // data_formart     数据格式, csv/libsvm
      // feat_sep         域分隔符
      // feat_values_sep  序列特征分隔符
      // feat_cfg         特征处理配置
      // mf               模型格式
      // im               模型地址
      // 如果是csv格式，必须在配置文件中设定csv_columns参数
      FmModel fm_model;
      int model_init_ret = fm_model.init("config/train.conf");
      if (0 != model_init_ret) {
        std::cout << " model init error : " << ret << std::endl;
      }
      ```

      2）每个线程可以创建自己的用于predict的instance

      ```c++
      FmPredictInstance* fm_instance = fm_model.getFmPredictInstance();
      ```

      3）调用预估

      ```c++
      // 方式1
      const char* intput_str =
          "123,aaa,信托|记账|酒店,2521,0.3,342|5212|839,24\n"
          "3423,bcd,培训|租房,342,1.2,44|3422|34|8,33\n";
      char predict_output[10240];
      fm_instance->fm_pred(intput_str, predict_output, sizeof(predict_output));
      
      // 方式2
      vector<string> input_vec;
      input_vec.push_back("3423,bcd,培训|租房,342,1.2,44|3422|34|8,33");
      input_vec.push_back("123,aaa,信托|记账|酒店,2521,0.3,342|5212|839,24");
      input_vec.push_back("3423,bcd,培训|租房,342,1.2,44|3422|34|8,33");
      
      vector<real_t> scores
      fm_instance->fm_pred(input_vec, scores);
      ```

      ###### java项目的调用方法

      可以通过jni调用以下几个c方法（同样声明在lib_fm_pred.h中）：

      ```c++
      extern "C" {
      
      //创建模型
      FmModel* fmModelCreate(const char* config_path);
      void fmModelRelease(FmModel* fm_model);
      
      
      // 每个线程创建自己的用于predict的instance
      FmPredictInstance * fmPredictInstanceCreate(FmModel* fm_model);
      void fmPredictInstanceRelease(FmPredictInstance* fm_instance);
      
      //调用预估
      /*
      @param input_str : support csv / libsvm formart
      @param output_str : output memory allocated by caller, will fill with scorelist joind by ','
      @param output_len : memory size of output_str
      @return: 0 : success;  other : faild
      */
      int fmPredict(FmPredictInstance * fm_instance, char* input_str, char* output_str, int output_len);
      
      }
      ```

3. #### 用于召回

   如果没有用到user与item的交叉特征，那么可以将该模型用于召回。根据FM的公式，可以将公式拆分为以下3个部分：

   1. user特征的一阶权重加和、所有user特征两两embedding点积之和。在线上预估时，该部分对所有item都一样，所以可以丢弃。
   2. item特征的一阶权重加和、所有item特征两两embedding点积之和。
   3. “user embedding加和” 与 “item embedding加和” 的点积。

   为了支持faiss检索，需要写成点积的形式，所以将向量改造，itemEmbedding为该item所有特征embedding加和，并增广1维，增广的这一维数值为 “所有Item特征一阶权重之和" + “所有Item特征隐向量两两点积之和”，线上召回时，先计算好userEmbedding，然后增广1维，增广的这一维数值为1。

   1. 离线：
      1. 每小时筛选出一部分适合召回的候选item（e.g.,过去7天至少被点击过3次的）。
      2. 针对每个候选item，提取其所有特征的一阶权重w和隐向量v，计算$$Item Embedding=concat(\sum w+\frac{1}{2}ReduceSum[(\sum v)^2-\sum v^2],\sum v)$$
      3. 将所有item embedding灌入FAISS建立索引。
   2. 在线：
      1. 用户请求到来时，提取其所有特征的隐向量v，计算$$User Embedding=concat(1,\sum v)$$
      2. 检索距离User Embedding最近的Top N item embedding。


特征处理配置文件格式规范

## 整体介绍

特征处理配置文件为一个json，包含3个属性，dense_features，sparse_features，varlen_sparse_features，分别为一个list，分别为对各个连续值特征、离散特征、 序列特征的处理方式。

## dense_features

连续特征将会被离散化。

这里支持几种典型的离散化方式。

1. 等宽离散化。这时候需要配置特征的最大值、最小值、等宽分桶的桶数。

   ```
         {
           "name": "dc1_2",
           "default_value": 0.0,
           "min_clip": 0.0,
           "max_clip": 1.0,
           "sparse_by_wide": 0.05
         }
   ```

2. 等频离散化。需要统计分割点，可以基于特征配置生成工具make_feat_config.py来自动生成等频离散化的分割点。或者手动指定分隔点，比如对于年龄特征基于年龄段做离散化。

   ```
   
   ```

3. 对数映射，然后按等宽分桶。 通常适用于点击UV等特征。

   ```
         {
           "name": "search_uv",
           "default_value": 0.0,
           "min_clip": 0.0,
           "max_clip": 15303.0,
           "add": 1.0,
           "log": 10,
           "sparse_by_wide": 1.0
         },
   ```

4. 指数映射，然后按等宽分桶。

   ```
         {
           "name": "bookWords",
           "default_value": 0.0,
           "min_clip": 0.0,
           "max_clip": 20000000.0,
           "add": 0.0,
           "multiply": 0.0001,
           "pow": 0.5,
           "sparse_by_wide": 1.0
         },
   ```
   
   
   
   

以下是详细的特征配置说明。注：当某个dense特征配置了以下多个操作属性时，操作顺序按照这里特征属性排列的顺序依次理。

| 特征属性名称     | 值类型        | 是否必须 | 说明                                                         | 示例                                                         |
| ---------------- | ------------- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| name             | string        | Y        | 特征名称                                                     | click_rate                                                   |
| default_value    | float         | Y        | 特征缺省值                                                   | 0.0                                                          |
| min_clip         | float         | Y        | 对特征原始值做最小值截断, if( x < min_clip) x = min_clip;    | 0.0                                                          |
| max_clip         | float         | Y        | 对特征原始值做最大值截断, if( x > max_clip) x = max_clip;    | 1.0                                                          |
| add              | float         | N        | 对特征取值加上一个数                                         | 3.0 则  x += 3.0                                             |
| multiply         | float         | N        | 对特征取值乘以一个数                                         | 3.0 则  x *= 3.0                                             |
| pow              | float         | N        | 以配置的值为幂，对特征取值按指数映射                         | 3.0 则  x = x^0.3                                            |
| log              | float         | N        | 以配置的值为base，对特征取值按对数映射                       | 3.0 则  x = log(x, 3.0)                                      |
| sparse_by_splits | 2D array of float | N        | 按指定的分隔值进行分桶。可以通过配置多个特征离散化分桶list支持多种离散化方式，从而将连续值映射为多个ID特征。每个元素都是升序float数组。<br /> <br />result = []<br />for plit_values in sparse_by_splits: <br />result.append(lower_bound(split_values, x))<br /><br />lower_bound函数为将x映射为split_values中第一个大于等于x的元素的下标。如果没有大于等于x的元素，则映射为最后一个元素下标。（同c++的std::lower_bound） | [[0, 0.1, 0.4, 1.0],  [0,  0.5, 1.0] ] <br />则： <br />当 x = -1 时, 将x映射为 0, 0 <br />当 x = 0 时, 将x映射为 0 当 x = 0.01  时, 将x映射为 1, 1 <br />当 x = 0.1  时, 将x映射为 1, 1<br />当 x = 0.2  时, 将x映射为  2, 1  当 x = 1.0  时, 将x映射为 3, 2 <br />当 x = 8.0  时, 将x映射为 3, 2 |
| sparse_by_wides   | array of float | N        | 以配置的值为分桶宽度，对特征进行等宽分桶<br />result = []<br />for wide in sparse_by_wides: <br />result.append(int (x / wide)) | [2.5, 5.0]<br />则：x = int(x / 2.5) , int(x/5.0)           |
| sparse_by_wide_bins_numbs | array of float | N | 以配置的值为分桶个数，对特征进行等宽分桶<br />result = []<br />for bins_num in sparse_by_wide_bins_numbs: <br />bins_wide = (max - min)/bins_num<br />result.append(int (x / bins_wide)) |  |

## sparse_features

离散特征都将被映射到一个整形ID，作为参数词典的下标。

default_id为0。当取不到特征时，直接映射为default_id，并忽略下面的特征操作。

unknown_id都为vocab_size-1（即最大有效ID+1）

mapping_type=="orig_id"时：小于0或者大于max_id的特征值将被映射为unknown_id。 

mapping_type=="dict"时：匹配不到特征ID词典的特征值将被映射为unknown_id。

mapping_type=="dynamic_dict" 时： 新ID都将被加入词典，所以不需要unknown_id。

mapping_type=="hash"时：所有的特征值都将被映射为合法的ID，所以不需要unknown_id。

| 特征属性名称          | 值类型 | 是否必须 | 说明                                                         | 示例            |
| --------------------- | ------ | -------- | ------------------------------------------------------------ | --------------- |
| name                  | string | Y        | 特征名称                                                     | "bid"           |
| value_type            | string | Y        | 原始值类型，取值有： "int32" "int64" "str"                   | str             |
| mapping_type          | string | Y        | 4个取值：<br />"orig_id" : 对原始值直接转无符号整形，如果原始值是字符串且不是合法的数字，则取default_id <br /> "hash" : 进行hash映射，hash方式统一采用MurmurHash3。<br />"dict" : 按词典进行映射，需要通过mapping_dict_name配置词典名称。<br />"dynamic_dict" : 边训练边构建词典，如果有配置mapping_dict_name则以其为初始化的词典。<br /><br />对于类别、渠道等不经常变动的离散特征，可以配置dict方式，按事先统计的映射表进行映射。<br />对于uid之类的大规模离散特征，可以配置hash以控制参数量。<br />对于待排序的itemID，通常不希望hash冲突，而且也有新老的更替，所以按hash或者dict进行映射可能都不理想，可以使用lru_dict，模型将一直维护最新的N个ID的参数。 | "dict"          |
| mapping_dict_name     | string | N        | 词典名称，通过该词典对原始值进行映射。 映射词典的key类型必须与x的值类型（value_type配置项）一致。value类型一律为32位无符号整形。 | "tag_name.dict" |
| max_id                | int    | N        | mapping_type=="orig_id"时需填写。                            | 8888            |
| vocab_size            | int    | N        | mapping_type=="dynamic_dict" 或者 mapping_type=="hash"时：确定参数的总个数，其他情况不需要。 |                 |
| shared_embedding_name | string | N        | 目前参数共享的实现还有问题。                                 |                 |


## varlen_sparse_features

包含sparse_features中的所有配置项，这里不再重复，仅列出varlen_sparse_features中多出的配置项。

| 征属性名称   | 值类型 | 是否必须 | 说明                    | 示例  |
| ------------ | ------ | -------- | ----------------------- | ----- |
| max_len      | int    | Y        | 序列的最大长度          | 20    |
| pooling_type | string | Y        | 目前只支持"sum" / "avg" | "sum" |


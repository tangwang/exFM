特征处理配置文件格式规范

## 整体介绍

特征处理配置文件为一个json，包含3个属性，dense_features，sparse_features，varlen_sparse_features，分别为一个list，分别为对各个连续值特征、离散特征、 序列特征的处理方式。

## dense_features

注：当某个dense特征配置了以下多个操作属性时，操作顺序按照这里特征属性排列的顺序依次理。

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
| sparse_by_splits | List of float | N        | 特征离散化分桶list，为升序float数组。 <br />对上述操作后的取值映射为该数组中第一个大于等于x的元素的下标。如果没有大于等于x的元素，则映射为最后一个元素下标。（同c++的std::lower_bound） | [0, 0.1,  0.4, 1.0] 则： 当 x = -1 时, 将x映射为 0 当 x = 0 时, 将x映射为 0 当 x = 0.01  时, 将x映射为 1 当 x = 0.1  时, 将x映射为 1 当 x = 1.0  时, 将x映射为 3 当 x = 8.0  时, 将x映射为 3 |
| sparse_by_wide   | float         | N        | 以配置的值为分桶宽度，对特征进行等宽分桶（<br />x = int (x / wide) | 2.5 则：x = int(x / 2.5)                                     |

## sparse_features

| 特征属性名称      | 值类型 | 是否必须 | 说明                                                         | 示例                 |
| ----------------- | ------ | -------- | ------------------------------------------------------------ | -------------------- |
| name              | string | Y        | 特征名称                                                     | "bid"                |
| value_type        | string | Y        | 原始值类型，取值有： "int32" "int64" "str"                   | str                  |
| default_id        | uint32 | Y        | 取值为[0, vocab_size-1]内的任意整数。 当取不到特征时，直接映射为default_id，并忽略以下操作。 | 0                    |
| mapping_type      | string | Y        | 三个取值：<br />"orig_id" : 对原始值直接转无符号整形，如果原始值是字符串且不是合法的数字，则取default_id <br />"dict" : 按词典进行映射，词典名称通过下方的配置项dict_name配置。<br /> "hash" : 进行hash映射，hash方式统一采用CityHash32（入参为字符串，如果原始值为整形则先转字符串，出参为无符号的32位整形）。 | "dict"               |
| mapping_dict_name | string | N        | 词典名称，通过该词典对原始值进行映射。 映射词典的key类型必须与x的值类型（value_type配置项）一致。value类型一律为32位无符号整形。 | "tagname2tagid.dict" |
| unknown_id        | int    | N        | 取值为[0, vocab_size-1]内的任意整数。 进行词典映射时，如果词典中没有匹配项，则映射到该ID。 | 8887                 |
| vocab_size        | int    | Y        | 取值必须大于0 在以上处理之后进行取模操作： x = x % vocab_size。 | 8888                 |


## varlen_sparse_features

相比于sparse_features中的特征配置多一个max_len配置项，代表序列的最大长度。


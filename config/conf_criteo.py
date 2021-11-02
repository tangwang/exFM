
# 训练数据格式，支持csv和libsvm，如果是csv格式必须确保第一行为表头（各列列名）,并且第一列为label
# label为0/1  或者-1/1，特征取值支持连续特征(float), 离散特征(int / string), 序列特征(list of int / string)
data_formart =  "csv"

# 如果是csv格式，通过csv_columns设定列名，或者设置csv_columns=[]，将输入数据的第一行读取为列名
csv_columns = []

# 对于csv和libsvm都需要配置域分隔符和序列特征中多个值的分隔符
feat_sep = ','
feat_values_sep = ';'

# dense特征
dense_feat_list = [f'I{i}' for i in range(1, 14)]

sparse_id_feat_list = []

varlen_sparse_id_feat_list = []

sparse_str_feat_list = [f'C{i}' for i in range(1, 27)] + [f'I{i}' for i in range(1, 14)]

varlen_sparse_str_feat_list = []


# 对于连续型特征等宽分桶的个数，可以配置多个分桶方式
dense_feat_wide_splits = [10, 25]
# 对于连续型特征等频分桶的个数，可以配置多个分桶方式
dense_feat_freq_splits = [10, 25]

default_value_of_dense_feat = 0

min_freq_for_sparse_feat_dict = 0 # 特征出现次数大于该值时，才加入特征ID映射词典

seq_feat_max_len = 30
seq_feat_pooling_type = "sum" # 暂时只支持sum和avg

# 配置离散特征的ID映射方式，支持dict/dynamic_dict/hash/orig_id
sparse_feat_mapping_type = "dynamic_dict"
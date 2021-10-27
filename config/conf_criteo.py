
# data_formart 训练数据格式，支持libsvm和csv两种格式
# 如果为libcsv格式，通过feat_sep, feat_kv_sep, feat_values_sep配置特征之间的分隔符、特征key和value之间的分隔符、varlen_sparse_feature中多个值的分隔符
# 如果为csv格式，通过feat_sep, feat_values_sep列之间的分隔符、varlen_sparse_feature中多个值的分隔符。将读取第一行作为表头，通过表头各列的名称与特征配置文件中的特征名关联。
# 分隔符如果需要配置为制表符、空格、等于号，分别用blank, tab, equal代替
data_formart =  "csv"

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

sparse_feat_mapping_type = "dict"
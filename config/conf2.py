
# data_formart 训练数据格式，支持libsvm和csv两种格式
# 如果为libcsv格式，通过feat_sep, feat_kv_sep, feat_values_sep配置特征之间的分隔符、特征key和value之间的分隔符、varlen_sparse_feature中多个值的分隔符
# 如果为csv格式，通过feat_sep, feat_values_sep列之间的分隔符、varlen_sparse_feature中多个值的分隔符。将读取第一行作为表头，通过表头各列的名称与特征配置文件中的特征名关联。
# 分隔符如果需要配置为制表符、空格、等于号，分别用blank, tab, equal代替
data_formart =  "csv"

feat_sep = ','
feat_values_sep = ';'

dense_feat_same_wide_bucket_numbers = [10, 25]
dense_feat_same_freq_bucket_numbers = [10, 25]

default_value_of_dense_feat = 0

# sparse特征ID词典，ID从2开始编号，0和1预留给default_id和unknown_id，分别代表没有值和不在词典中的值
default_id_of_sparse_feat = 0
unknown_id_of_sparse_feat = 1
min_freq_for_sparse_feat_dict = 0 # 特征出现大于该值时，才加入特征ID映射词典

seq_feat_pooling_type = "sum" # 暂时只支持sum和avg


# dense特征
dense_feat_list = [f'I{i}' for i in range(1, 14)]

sparse_id_feat_list = []

varlen_sparse_id_feat_list = []

sparse_str_feat_list = [f'C{i}' for i in range(1, 27)] + [f'I{i}' for i in range(1, 14)]

varlen_sparse_str_feat_list = []


# 训练数据格式，支持csv和libsvm
# label>0为正样本，否则视为负样本。特征取值支持连续特征(float), 离散特征(int / string), 序列特征(list of int / string)
data_formart =  "csv"

# 如果是csv格式，通过csv_columns设定列名，或者设置csv_columns=[]，将输入数据的第一行读取为列名
csv_columns = ['label', 'item_id', 'chanel', 'item_tags', 'item_clicks', 'item_price', 'user_click_list', 'user_age']

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
sparse_feat_mapping_type = "dynamic_dict"
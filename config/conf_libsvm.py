
# 训练数据格式，支持csv和libsvm，如果是csv格式必须确保第一行为表头（各列列名）并且第一列为label。label取值为0/1或者-1/1
data_formart     =  "libsvm"

# 对于csv和libsvm都需要配置域分隔符和序列特征中多个值的分隔符
feat_sep = '\t'
feat_values_sep = ','


# dense特征
dense_feat_list = [
'bc_new'                 ,
'bookChapters'           ,
'bookWords'              ,
'dBkW1'                  ,
'deTsW1'                 ,
'epubPro'                ,
'fans_1'                 ,
'fans_2'                 ,
'fans_3'                 ,
'fans_4'                 ,
'fans_5'                 ,
'gc_new'                 ,
'iDsYT'                  ,
'pop'                    ,
'pr'                     ,
'Q_pr'                   ,
'sales_d1'               ,
'sales_volume_down'      ,
'search_uv'              
]

# sparse 特征，取值按照int解析
sparse_id_feat_list = [
'c2'            , 
'c3'            , 
'authorId'      , 
'pubId'         , 
'c1'            , 
'bid'           , 
'isVip'         , 
'mode_flag'     , 
]

# varlen sparse 特征，取值按照int解析
varlen_sparse_id_feat_list = [
'L_authorId_his'    ,
'L_bid_his'         ,
'L_c1_his'          ,
'L_c2_his'          ,
'L_c3_his'          ,
'L_pubId_his'       ,
'L_tag_his'         ,
'tag'              ,
]


# sparse 特征，取值按照字符串解析
#sparse_str_feat_list = ['c2']
sparse_str_feat_list = []

# varlen sparse 特征，取值按照字符串解析
#varlen_sparse_str_feat_list = ['L_authorId_his' ]
varlen_sparse_str_feat_list = []


dense_feat_wide_splits = [10, 25]
dense_feat_freq_splits = [10, 25]

default_value_of_dense_feat = 0

min_freq_for_sparse_feat_dict = 0 # 特征出现次数大于该值时，才加入特征ID映射词典

seq_feat_max_len = 30
seq_feat_pooling_type = "sum" # 暂时只支持sum和avg

# 配置离散特征的ID映射方式，支持dict/dynamic_dict/hash/orig_id
sparse_feat_mapping_type = "dynamic_dict"
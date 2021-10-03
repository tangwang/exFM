
field_split = '\t'
kv_seperator = ':'
value_list_split = ','

dense_fea_same_wide_bucket_numbers = [10, 25]
dense_fea_same_freq_bucket_numbers = [10, 25]

default_value_of_dense_feat = 0

# sparse特征ID词典，ID从2开始编号，0和1预留给default_id和unknown_id，分别代表没有值和不在词典中的值
default_id_of_sparse_feat = 0
unknown_id_of_sparse_feat = 1


# dense特征
dense_fea_list = set([
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
])

# sparse 特征，取值按照int解析
sparse_id_fea_list = set([
'c2'            , 
'c3'            , 
'authorId'      , 
'pubId'         , 
'c1'            , 
'bid'           , 
'isVip'         , 
'mode_flag'     , 
])

# varlen sparse 特征，取值按照int解析
varlen_sparse_id_fea_list = set([
'L_authorId_his'    ,
'L_bid_his'         ,
'L_c1_his'          ,
'L_c2_his'          ,
'L_c3_his'          ,
'L_pubId_his'       ,
'L_tag_his'         ,
'tag'              ,
])


# sparse 特征，取值按照字符串解析
#sparse_str_fea_list = ['c2']
sparse_str_fea_list = []

# varlen sparse 特征，取值按照字符串解析
#varlen_sparse_str_fea_list = ['L_authorId_his' ]
varlen_sparse_str_fea_list = []


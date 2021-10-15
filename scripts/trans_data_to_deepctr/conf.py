
denseFeaList = [
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

# (max_id, embed_size )
sparseFeaList = [
'c2'            , 
'c3'            , 
'authorId'      , 
'pubId'         , 
'c1'            , 
'bid'           , 
'isVip'        , 
'mode_flag'    
]

# value为最大长度
varlenSparseFeaList = {
'L_authorId_his'    :  30  ,
'L_bid_his'         :  30  ,
'L_c1_his'          :  30  ,
'L_c2_his'          :  30  ,
'L_c3_his'          :  30  ,
'L_pubId_his'       :  30  ,
'L_tag_his'         :  30  ,
'tag'               :  4  
}


feat_type_dense = 1
feat_type_sparce = 2
feat_type_varlen_sparce = 3



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

# (max_id, embed_size, max_len )
seqFeaList = {
'L_authorId_his'    : (0, 16, 30 ) ,
'L_bid_his'         : (0, 16, 30 ) ,
'L_c1_his'          : (0, 8, 30 ) ,
'L_c2_his'          : (0, 8, 30 ) ,
'L_c3_his'          : (0, 8, 30 ) ,
'L_pubId_his'       : (0, 8, 30 ) ,
'L_tag_his'         : (0, 8, 30 ) ,
'tag'               : (0, 8, 4  ),
}

# (max_id, embed_size )
sparseFeaList = {
'c2'            : (0, 8)  , 
'c3'            : (0, 8)  , 
'authorId'      : (0, 8)  , 
'pubId'         : (0, 8)  , 
'c1'            : (0, 4)  , 
'bid'           : (0, 16)  , 
'isVip'         : (0, 1)  , 
'mode_flag'     : (0, 1)  , 
}

fea_type_dense = 1
fea_type_sparce = 2
fea_type_seq = 3


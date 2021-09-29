import sys
import logging
import traceback
import cgitb
import os
import json
import argparse
from fea_statis_conf import *
import numpy as np
#from collections import OrderedDict
from collections import Counter

dense_fea_dict = {k : [] for k in dense_fea_list}
sparse_id_fea_dict = {k : [] for k in sparse_id_fea_list}
varlen_sparse_id_fea_dict = {k : [] for k in varlen_sparse_id_fea_list}
varlen_sparse_id_fea_len_dict = {k : [] for k in varlen_sparse_id_fea_list}
sparse_str_fea_dict = {k : [] for k in sparse_str_fea_list}
varlen_sparse_str_fea_dict = {k : [] for k in varlen_sparse_str_fea_list}
varlen_sparse_str_fea_len_dict = {k : [] for k in varlen_sparse_str_fea_list}

def parse_id(x):
    if not x.isdigit():
        if (x.endswith('.0')):
            return int(float(x))
        print(f'not digit: {x}', file=sys.stderr)
        return -1
    return int(x)

def save_fea_id_mapping_dict(mapping_dict_name, fea_num_statis_dict_path, fea_name, fea_id_list):
    sorted_fea_num_list = sorted(fea_id_list.items(), key=lambda x:x[1], reverse=True)
    with open(mapping_dict_name, 'w') as f_id_mapping:
        with open(fea_num_statis_dict_path, 'w') as f_num_statis:
            for idx, (k, v) in enumerate(sorted_fea_num_list):
                # id 0和1 预留出来，0可以用来做default，1可以用来做unknown
                print(k, idx+2, sep=' ', file=f_id_mapping)
                print(k, v, sep=' ', file=f_num_statis)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str)
    args = parser.parse_args()

    id_map_dict_path = os.path.join(args.output_path, 'fea_id_mapping_dict')
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(id_map_dict_path, exist_ok=True)

    ##########################################
    # 解析数据
    ##########################################
    for line in sys.stdin:
        segs = line.rstrip().split(field_split)[1:]

        for kv in segs:
            try:
                k, v = kv.split(kv_seperator)
                if not v:
                    continue

                if k in dense_fea_dict:
                    dense_fea_dict[k].append(float(v))
                elif k in sparse_id_fea_dict:
                    # 兼容有多个值的情况
                    if value_list_split in v:
                        v = v.split(value_list_split)[0]
                    sparse_id_fea_dict[k].append(parse_id(v) if '.' in v else int(v))
                
                elif k in varlen_sparse_id_fea_dict:
                    #varlen_sparse_id_fea_dict[k].append([int(x) for x in v.split(value_list_split)])
                    value_list = [parse_id(x) for x in v.split(value_list_split)]
                    varlen_sparse_id_fea_dict[k].extend(value_list)
                    varlen_sparse_id_fea_len_dict[k].append(len(value_list))

                elif k in sparse_str_fea_dict:
                    # 兼容有多个值的情况
                    if value_list_split in v:
                        v = v.split(value_list_split)[0]
                    sparse_id_fea_dict[k].append(v)
                
                elif k in varlen_sparse_str_fea_dict:
                    #varlen_sparse_id_fea_dict[k].append([int(x) for x in v.split(value_list_split)])
                    value_list = v.split(value_list_split)
                    varlen_sparse_id_fea_dict[k].extend(value_list)
                    varlen_sparse_id_fea_len_dict[k].append(len(value_list))

            except Exception as e:
                print('exception occured while proc ', kv, file=sys.stderr)
                traceback.print_exc()


    ##########################################
    # 打印统计信息，输出特征配置文件
    ##########################################

    dense_features = []
    sparse_features = []
    varlen_sparse_features = []
    
    for k, v in dense_fea_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'fea {k} is empty')
            continue
        v = np.array(sorted(v), dtype=np.float)
        min_v = float(np.min(v))
        max_v = float(np.max(v))
        mean = float(np.mean(v))
        std = np.std(v)
        num = len(v)
        percentile_values = []
        for n in dense_fea_same_freq_bucket_numbers:
            percentile_values.append([np.percentile(v, 100*pos/n) for pos in range(1, n)])
        print(f' dense_fea: {k} ')
        print('min_clip', min_v)
        print('max_clip', max_v)
        print('mean', mean)
        print('std', std)
        print('num', num)
        print('percentile_values', percentile_values)

        dense_features.append( {'name' : k, 
                'min_clip' : min_v,
                 'max_clip' : max_v,
                 'default_value' : default_dense_value,
                 'sparse_by_wide' : dense_fea_same_wide_bucket_numbers,
                'sparse_by_splits' : percentile_values,
                } )

    for k, v in sparse_id_fea_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'fea {k} is empty')
            continue
        v = np.array(v, dtype=np.int32)
        min_id = int(np.min(v))
        max_id = int(np.max(v))
        num = len(v)
        ids_num = len(set(v))
        print(f' sparse_fea : {k} ')
        print('min_id', min_id)
        print('max_id', max_id)
        print('num', num)
        print('ids_num', ids_num)

        if ids_num / max_id < use_hash_sparsity_threshold:
            use_hash = True
            vocab_size = int(2 * ids_num)
        else:
            use_hash = False
            vocab_size = max_id

        mapping_dict_name = os.path.join(id_map_dict_path, f'{k}.dict')
        fea_num_statis_dict_path = os.path.join(id_map_dict_path, f'num_static.{k}.dict')
        fea_counts = Counter(v)
        save_fea_id_mapping_dict(mapping_dict_name, fea_num_statis_dict_path, k, fea_counts)

        sparse_features.append( {'name' : k, 
                'vocab_size' : vocab_size,
                'max_id' : max_id,
                'use_id_mapping' : 0 if (max_id < 1000 or max_id*3 > ids_num) else 1,
                'use_hash' : use_hash,
                'mapping_dict_name' : mapping_dict_name,
                 'shared_embedding_name' : '',
                 'default_id' : default_sparse_value
                } )
        
    for k, v in varlen_sparse_id_fea_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'fea {k} is empty')
            continue
        v = np.array(sorted(v), dtype=np.int32)
        len_list = np.array(varlen_sparse_id_fea_len_dict[k], dtype=np.int32)
        min_id = int(np.min(v))
        max_id = int(np.max(v))
        mean_len = int(np.mean(len_list))
        max_len = int(np.max(len_list))
        num = len(len_list)
        ids_num = len(set(v))
        print(f' varlen_sparse_fea : {k} ')
        print('min_id', min_id)
        print('max_id', max_id)
        print('mean_len', mean_len)
        print('max_len', max_len)
        print('num', num)
        print('ids_num', ids_num)

        if ids_num / max_id < use_hash_sparsity_threshold:
            use_hash = True
            vocab_size = int(2 * ids_num)
        else:
            use_hash = False
            vocab_size = max_id

        mapping_dict_name = os.path.join(id_map_dict_path, f'{k}.dict')
        fea_num_statis_dict_path = os.path.join(id_map_dict_path, f'num_static.{k}.dict')
        fea_counts = Counter(v)
        save_fea_id_mapping_dict(mapping_dict_name, fea_num_statis_dict_path, k, fea_counts)

        varlen_sparse_features.append( {'name' : k,
                'vocab_size' : vocab_size,
                'max_id' : max_id,
                'use_id_mapping' : 0 if (max_id < 1000 or max_id*3 > ids_num) else 1,
                'use_hash' : use_hash,
                 'default_id' : default_sparse_value,
                 'max_len' : max_len,
                 'mapping_dict_name' : mapping_dict_name,
                 'shared_embedding_name' : '',
                'pooling_type' : 'sum'
                } )

    for k, v in sparse_str_fea_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'fea {k} is empty')
            continue
        num = len(v)
        ids_num = len(set(v))
        print(f' sparse_str_fea : {k} ')
        print('num', num)
        print('ids_num', ids_num)

        use_hash = False
        vocab_size = ids_num

        sparse_features.append( {'name' : k, 
                'vocab_size' : vocab_size,
                'max_id' : max_id,
                'use_id_mapping' : 0 if (max_id < 1000 or max_id*3 > ids_num) else 1,
                'use_hash' : use_hash,
                'mapping_dict_name' : mapping_dict_name,
                 'shared_embedding_name' : '',
                 'default_id' : default_sparse_value
                } )
        
    for k, v in varlen_sparse_str_fea_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'fea {k} is empty')
            continue
        len_list = np.array(varlen_sparse_id_fea_len_dict[k], dtype=np.int32)
        mean_len = int(np.mean(len_list))
        max_len = int(np.max(len_list))
        num = len(len_list)
        ids_num = len(set(v))
        print(f' varlen_str_sparse_fea : {k} ')
        print('mean_len', mean_len)
        print('max_len', max_len)
        print('num', num)
        print('ids_num', ids_num)

        use_hash = False
        vocab_size = ids_num

        mapping_dict_name = os.path.join(id_map_dict_path, f'{k}.dict')
        fea_num_statis_dict_path = os.path.join(id_map_dict_path, f'num_static.{k}.dict')

        fea_counts = Counter(v)
        save_fea_id_mapping_dict(mapping_dict_name, fea_num_statis_dict_path, k, fea_counts)

        varlen_sparse_features.append( {'name' : k,
                'vocab_size' : vocab_size,
                'max_id' : max_id,
                'use_id_mapping' : 0 if (max_id < 1000 or max_id*3 > ids_num) else 1,
                'use_hash' : use_hash,
                'mapping_dict_name' : mapping_dict_name,
                 'shared_embedding_name' : '',
                 'default_id' : default_sparse_value,
                 'max_len' : max_len,
                 'pooling_type' : seq_pooling_type
                } )

    with open(os.path.join(args.output_path, 'feature_config.json'), 'w') as f:
        fea_config = {"dense_features" : dense_features, "sparse_features" : sparse_features, "varlen_sparse_features" : varlen_sparse_features}
        fea_config_str = json.dumps(fea_config, indent=4) # sort_keys=True, 
        f.write(fea_config_str)
        


import sys
import traceback
import cgitb
import os
import json
import argparse
import importlib
from numpy.lib.shape_base import column_stack
import numpy as np
#from collections import OrderedDict
from collections import Counter
import multiprocessing

from conf import *

max_hash_buckets = 10000000
min_hash_buckets = 200
feat_kv_sep = ':'

def proc_line__csv(job_queue, process_id, dict_feat_name_to_id,
    a_dense_feat_dict,
    a_sparse_id_feat_dict,
    a_varlen_sparse_id_feat_dict,
    a_varlen_sparse_id_feat_len_dict,
    a_sparse_str_feat_dict,
    a_varlen_sparse_str_feat_dict,
    a_varlen_sparse_str_feat_len_dict):

    print('processor ', process_id, ' started.')

    max_column_id = max(dict_feat_name_to_id.values())
    try:
        dense_feat_dict = {dict_feat_name_to_id[k] : [] for k in dense_feat_list}
        sparse_id_feat_dict = {dict_feat_name_to_id[k] : [] for k in sparse_id_feat_list}
        varlen_sparse_id_feat_dict = {dict_feat_name_to_id[k] : [] for k in varlen_sparse_id_feat_list}
        varlen_sparse_id_feat_len_dict = {dict_feat_name_to_id[k] : [] for k in varlen_sparse_id_feat_list}
        sparse_str_feat_dict = {dict_feat_name_to_id[k] : [] for k in sparse_str_feat_list}
        varlen_sparse_str_feat_dict = {dict_feat_name_to_id[k] : [] for k in varlen_sparse_str_feat_list}
        varlen_sparse_str_feat_len_dict = {dict_feat_name_to_id[k] : [] for k in varlen_sparse_str_feat_list}
    except Exception as e:
        print('error: feature names not match with column names. exception: ', e)
        return

    while True:
        line = job_queue.get()
        if line is None:
            break
        segs = line.rstrip('\n').split(feat_sep)
        if len(segs) < max_column_id + 1:
            continue

        for column_id, column_list in dense_feat_dict.items():
            column_list.append(float(segs[column_id]) if segs[column_id] else 0.0)
        for column_id, column_list in sparse_id_feat_dict.items():
            # 兼容有多个值的情况
            v = segs[column_id]
            if not v:
                continue
            if feat_values_sep in v:
                v = v.split(feat_values_sep)[0]
            column_list.append(parse_id(v) if '.' in v else int(v))
        for column_id, column_list in varlen_sparse_id_feat_dict.items():
            v = segs[column_id]
            if not v:
                continue
            value_list = [parse_id(x) for x in v.split(feat_values_sep)]
            column_list.extend(value_list)
            varlen_sparse_id_feat_len_dict[column_id].append(len(value_list))

        for column_id, column_list in sparse_str_feat_dict.items():
            v = segs[column_id]
            if not v:
                continue
            # 兼容有多个值的情况
            if feat_values_sep in v:
                v = v.split(feat_values_sep)[0]
            column_list.append(v)

        for column_id, column_list in varlen_sparse_str_feat_dict.items():
            v = segs[column_id]
            if not v:
                continue
            value_list = [parse_id(x) for x in v.split(feat_values_sep)]
            column_list.extend(value_list)
            varlen_sparse_str_feat_len_dict[column_id].append(len(value_list))

    dict_feat_id_to_name = dict((v, k) for k, v in dict_feat_name_to_id.items())
    dense_feat_dict                   = dict((dict_feat_id_to_name[k], v) for k, v in dense_feat_dict                .items())
    sparse_id_feat_dict               = dict((dict_feat_id_to_name[k], v) for k, v in sparse_id_feat_dict            .items())
    varlen_sparse_id_feat_dict        = dict((dict_feat_id_to_name[k], v) for k, v in varlen_sparse_id_feat_dict     .items())
    varlen_sparse_id_feat_len_dict    = dict((dict_feat_id_to_name[k], v) for k, v in varlen_sparse_id_feat_len_dict .items())
    sparse_str_feat_dict              = dict((dict_feat_id_to_name[k], v) for k, v in sparse_str_feat_dict           .items())
    varlen_sparse_str_feat_dict       = dict((dict_feat_id_to_name[k], v) for k, v in varlen_sparse_str_feat_dict    .items())
    varlen_sparse_str_feat_len_dict   = dict((dict_feat_id_to_name[k], v) for k, v in varlen_sparse_str_feat_len_dict.items())

    a_dense_feat_dict                   .update(dense_feat_dict                )
    a_sparse_id_feat_dict               .update(sparse_id_feat_dict            )
    a_varlen_sparse_id_feat_dict        .update(varlen_sparse_id_feat_dict     )
    a_varlen_sparse_id_feat_len_dict    .update(varlen_sparse_id_feat_len_dict )
    a_sparse_str_feat_dict              .update(sparse_str_feat_dict           )
    a_varlen_sparse_str_feat_dict       .update(varlen_sparse_str_feat_dict    )
    a_varlen_sparse_str_feat_len_dict   .update(varlen_sparse_str_feat_len_dict)


def proc_line__libsvm(job_queue, process_id, 
    a_dense_feat_dict,
    a_sparse_id_feat_dict,
    a_varlen_sparse_id_feat_dict,
    a_varlen_sparse_id_feat_len_dict,
    a_sparse_str_feat_dict,
    a_varlen_sparse_str_feat_dict,
    a_varlen_sparse_str_feat_len_dict):

    print('processor ', process_id, ' started.')

    dense_feat_dict = {k : [] for k in dense_feat_list}
    sparse_id_feat_dict = {k : [] for k in sparse_id_feat_list}
    varlen_sparse_id_feat_dict = {k : [] for k in varlen_sparse_id_feat_list}
    varlen_sparse_id_feat_len_dict = {k : [] for k in varlen_sparse_id_feat_list}
    sparse_str_feat_dict = {k : [] for k in sparse_str_feat_list}
    varlen_sparse_str_feat_dict = {k : [] for k in varlen_sparse_str_feat_list}
    varlen_sparse_str_feat_len_dict = {k : [] for k in varlen_sparse_str_feat_list}

    while True:
        line = job_queue.get()
        if line is None:
            break
        segs = line.rstrip('\n').split(feat_sep)[1:]

        for kv in segs:
            k, v = kv.split(feat_kv_sep)
            if not v:
                continue

            if k in dense_feat_dict:
                dense_feat_dict[k].append(float(v))
            elif k in sparse_id_feat_dict:
                # 兼容有多个值的情况
                if feat_values_sep in v:
                    v = v.split(feat_values_sep)[0]
                sparse_id_feat_dict[k].append(parse_id(v) if '.' in v else int(v))
            
            elif k in varlen_sparse_id_feat_dict:
                #varlen_sparse_id_feat_dict[k].append([int(x) for x in v.split(feat_values_sep)])
                value_list = [parse_id(x) for x in v.split(feat_values_sep)]
                varlen_sparse_id_feat_dict[k].extend(value_list)
                varlen_sparse_id_feat_len_dict[k].append(len(value_list))

            elif k in sparse_str_feat_dict:
                # 兼容有多个值的情况
                if feat_values_sep in v:
                    v = v.split(feat_values_sep)[0]
                sparse_id_feat_dict[k].append(v)
            
            elif k in varlen_sparse_str_feat_dict:
                #varlen_sparse_id_feat_dict[k].append([int(x) for x in v.split(feat_values_sep)])
                value_list = v.split(feat_values_sep)
                varlen_sparse_str_feat_dict[k].extend(value_list)
                varlen_sparse_str_feat_len_dict[k].append(len(value_list))

    a_dense_feat_dict                   .update(dense_feat_dict                )
    a_sparse_id_feat_dict               .update(sparse_id_feat_dict            )
    a_varlen_sparse_id_feat_dict        .update(varlen_sparse_id_feat_dict     )
    a_varlen_sparse_id_feat_len_dict    .update(varlen_sparse_id_feat_len_dict )
    a_sparse_str_feat_dict              .update(sparse_str_feat_dict           )
    a_varlen_sparse_str_feat_dict       .update(varlen_sparse_str_feat_dict    )
    a_varlen_sparse_str_feat_len_dict   .update(varlen_sparse_str_feat_len_dict)


def merge_dicts_to(list_of_dict, d):
    for dict_i in list_of_dict:
        for k, v in dict_i.items():
            d.setdefault(k, []).extend(v)
        
def parse_id(x):
    if not x.isdigit():
        # 暂时特殊处理，以后不需要兼容float类型数据
        try:
            return int(round(float(x)))
        except Exception as e:
            print(f'not digit: {x}', file=sys.stderr)
            pass
        return -1
    return int(x)


def save_feat_id_dict(mapping_dict_path, feat_num_statis_dict_path, feat_name, feat_id_list):
    sorted_feat_num_list = sorted(feat_id_list.items(), key=lambda x:x[1], reverse=True)
    with open(mapping_dict_path, 'w') as f_id_mapping:
        with open(feat_num_statis_dict_path, 'w') as f_num_statis:
            for idx, (k, v) in enumerate(sorted_feat_num_list):
                # id 0和1 预留出来，0可以用来做default，1可以用来做unknown
                if v > min_freq_for_sparse_feat_dict:
                    print(k, idx+2, sep=' ', file=f_id_mapping)
                print(k, v, sep=' ', file=f_num_statis)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_path', type=str)
    parser.add_argument('--cpu_num', default=10, type=int)
    args = parser.parse_args()

    id_map_dict_path = os.path.join(args.output_path, 'feat_id_dict')
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(id_map_dict_path, exist_ok=True)

    ##########################################
    # 解析数据
    ##########################################
    cpu_num = args.cpu_num
    manager = multiprocessing.Manager()
    #job_queue = manager.Queue(maxsize = 500000)
    job_queues = [manager.Queue(maxsize = 20000) for i in range(cpu_num)]
    list_of__dense_feat_dict                   = [manager.dict() for i in range(cpu_num)]
    list_of__sparse_id_feat_dict               = [manager.dict() for i in range(cpu_num)]
    list_of__varlen_sparse_id_feat_dict        = [manager.dict() for i in range(cpu_num)]
    list_of__varlen_sparse_id_feat_len_dict    = [manager.dict() for i in range(cpu_num)]
    list_of__sparse_str_feat_dict              = [manager.dict() for i in range(cpu_num)]
    list_of__varlen_sparse_str_feat_dict       = [manager.dict() for i in range(cpu_num)]
    list_of__varlen_sparse_str_feat_len_dict   = [manager.dict() for i in range(cpu_num)]

    pool = multiprocessing.Pool(cpu_num + 1)

    workers = []
    if data_formart == 'csv':
        if len(csv_columns) == 0:
            csv_columns = sys.stdin.readline().rstrip('\n').split(feat_sep)
        dict_feat_name_to_id = dict((v, j) for j,v in enumerate(csv_columns))
        for i in range(cpu_num):
            worker = pool.apply_async(proc_line__csv, (job_queues[i], i, dict_feat_name_to_id, 
                list_of__dense_feat_dict                [i],
                list_of__sparse_id_feat_dict            [i],
                list_of__varlen_sparse_id_feat_dict     [i],
                list_of__varlen_sparse_id_feat_len_dict [i],
                list_of__sparse_str_feat_dict           [i],
                list_of__varlen_sparse_str_feat_dict    [i],
                list_of__varlen_sparse_str_feat_len_dict[i],
            ))
            workers.append(worker)
    elif data_formart == 'libsvm':
        for i in range(cpu_num):
            worker = pool.apply_async(proc_line__libsvm, (job_queues[i], i, 
                list_of__dense_feat_dict                [i],
                list_of__sparse_id_feat_dict            [i],
                list_of__varlen_sparse_id_feat_dict     [i],
                list_of__varlen_sparse_id_feat_len_dict [i],
                list_of__sparse_str_feat_dict           [i],
                list_of__varlen_sparse_str_feat_dict    [i],
                list_of__varlen_sparse_str_feat_len_dict[i],
            ))
            workers.append(worker)
    else:
        print('data_formart must be libsvm / csv. exit..')
        exit(1)

    idx = 0
    for line in sys.stdin:
        job_queues[idx % cpu_num].put(line)
        idx += 1
    for i in range(cpu_num):
        job_queues[i].put(None)

    for worker in workers:
        worker.get()

    dense_feat_dict                  = {}
    sparse_id_feat_dict              = {}
    varlen_sparse_id_feat_dict       = {}
    varlen_sparse_id_feat_len_dict   = {}
    sparse_str_feat_dict             = {}
    varlen_sparse_str_feat_dict      = {}
    varlen_sparse_str_feat_len_dict  = {}

    merge_dicts_to(list_of__dense_feat_dict                 , dense_feat_dict                  )
    merge_dicts_to(list_of__sparse_id_feat_dict             , sparse_id_feat_dict              )
    merge_dicts_to(list_of__varlen_sparse_id_feat_dict      , varlen_sparse_id_feat_dict       )
    merge_dicts_to(list_of__varlen_sparse_id_feat_len_dict  , varlen_sparse_id_feat_len_dict   )
    merge_dicts_to(list_of__sparse_str_feat_dict            , sparse_str_feat_dict             )
    merge_dicts_to(list_of__varlen_sparse_str_feat_dict     , varlen_sparse_str_feat_dict      )
    merge_dicts_to(list_of__varlen_sparse_str_feat_len_dict , varlen_sparse_str_feat_len_dict  )


    ##########################################
    # 打印统计信息，输出特征配置文件
    ##########################################

    dense_features = []
    sparse_features = []
    varlen_sparse_features = []
    
    for k, v in dense_feat_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'feat {k} is empty')
            continue
        v = np.array(sorted(v), dtype=np.float)
        min_v = float(np.min(v))
        max_v = float(np.max(v))
        mean = float(np.mean(v))
        std = np.std(v)
        num = len(v)
        percentile_values = []
        for n in dense_feat_freq_splits:
            split_values = [np.percentile(v, 100*pos/n) for pos in range(1, n)]
            compact_split_values = [split_values[0]]
            last_value = split_values[0]
            for split_value in split_values[1:]:
                if split_value - last_value > 1e-3:
                    compact_split_values.append(split_value)
                    last_value = split_value
            percentile_values.append(compact_split_values)
        print(f' dense_feat: {k} ')
        print('min_clip', min_v)
        print('max_clip', max_v)
        print('mean', mean)
        print('std', std)
        print('num', num)
        print('percentile_values', percentile_values)

        dense_features.append( {'name' : k, 
                'min_clip' : min_v,
                 'max_clip' : max_v,
                 'default_value' : default_value_of_dense_feat,
                 'sparse_by_wide_bins_numbs' : dense_feat_wide_splits,
                'sparse_by_splits' : percentile_values,
                } )

    for k, v in sparse_id_feat_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'feat {k} is empty')
            continue
        v = np.array(v, dtype=np.int32)
        min_id = int(np.min(v))
        max_id = int(np.max(v))
        num = len(v)
        ids_num = len(set(v))
        vocab_size = min(ids_num + 5000, 10 * (ids_num+10))
        if (vocab_size < min_hash_buckets):
            vocab_size = min_hash_buckets
        if (vocab_size > max_hash_buckets):
            vocab_size = max_hash_buckets
        print(f' sparse_feat : {k} ')
        print('min_id', min_id)
        print('max_id', max_id)
        print('num', num)
        print('ids_num', ids_num)
        print('vocab_size', vocab_size)

        mapping_dict_name = f'{k}.dict'
        mapping_dict_path = os.path.join(id_map_dict_path, mapping_dict_name)
        feat_num_statis_dict_path = os.path.join(id_map_dict_path, f'num_static.{k}.dict')
        feat_counts = Counter(v)
        save_feat_id_dict(mapping_dict_path, feat_num_statis_dict_path, k, feat_counts)

        sparse_features.append( {'name' : k, 
                'max_id' : max_id,
                'ids_num' : ids_num,
                'vocab_size' : vocab_size,
                'value_type' : 'int64',
                "mapping_type" : sparse_feat_mapping_type,
                'mapping_dict_name' : mapping_dict_name,
                 'shared_embedding_name' : ''
                } )
        
    for k, v in varlen_sparse_id_feat_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'feat {k} is empty')
            continue
        v = np.array(sorted(v), dtype=np.int32)
        len_list = np.array(varlen_sparse_id_feat_len_dict[k], dtype=np.int32)
        min_id = int(np.min(v))
        max_id = int(np.max(v))
        mean_len = int(np.mean(len_list))
        max_len = int(np.max(len_list))
        num = len(len_list)
        ids_num = len(set(v))
        vocab_size = min(ids_num + 5000, 10 * (ids_num+10))
        if (vocab_size < min_hash_buckets):
            vocab_size = min_hash_buckets
        if (vocab_size > max_hash_buckets):
            vocab_size = max_hash_buckets
        print(f' varlen_sparse_feat : {k} ')
        print('min_id', min_id)
        print('max_id', max_id)
        print('mean_len', mean_len)
        print('max_len', max_len)
        print('num', num)
        print('ids_num', ids_num)
        print('vocab_size', vocab_size)

        mapping_dict_name = f'{k}.dict'
        mapping_dict_path = os.path.join(id_map_dict_path, mapping_dict_name)
        feat_num_statis_dict_path = os.path.join(id_map_dict_path, f'num_static.{k}.dict')
        feat_counts = Counter(v)
        save_feat_id_dict(mapping_dict_path, feat_num_statis_dict_path, k, feat_counts)

        varlen_sparse_features.append( {'name' : k,
                'max_id' : max_id,
                'ids_num' : ids_num,
                'vocab_size' : vocab_size,
                'value_type' : 'int64',
                "mapping_type" : sparse_feat_mapping_type,
                'max_len' : min(seq_feat_max_len, max_len),
                'pooling_type' : seq_feat_pooling_type,
                'mapping_dict_name' : mapping_dict_name,
                'shared_embedding_name' : ''
                } )

    for k, v in sparse_str_feat_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'feat {k} is empty')
            continue
        num = len(v)
        ids_num = len(set(v))
        vocab_size = min(ids_num + 5000, 10 * (ids_num+10))
        if (vocab_size < min_hash_buckets):
            vocab_size = min_hash_buckets
        if (vocab_size > max_hash_buckets):
            vocab_size = max_hash_buckets
        print(f' sparse_str_feat : {k} ')
        print('num', num)
        print('ids_num', ids_num)
        print('vocab_size', vocab_size)

        mapping_dict_name = f'{k}.dict'
        mapping_dict_path = os.path.join(id_map_dict_path, mapping_dict_name)
        feat_num_statis_dict_path = os.path.join(id_map_dict_path, f'num_static.{k}.dict')
        feat_counts = Counter(v)
        save_feat_id_dict(mapping_dict_path, feat_num_statis_dict_path, k, feat_counts)

        sparse_features.append( {'name' : k, 
                'value_type' : 'str',
                 "mapping_type" : sparse_feat_mapping_type,
                'ids_num' : ids_num,
                'vocab_size' : vocab_size,
                'mapping_dict_name' : mapping_dict_name,
                 'shared_embedding_name' : ''
                } )
        
    for k, v in varlen_sparse_str_feat_dict.items():
        print('\n------------------------------------------')
        if len(v) == 0:
            print(f'feat {k} is empty')
            continue
        len_list = np.array(varlen_sparse_str_feat_len_dict[k], dtype=np.int32)
        mean_len = int(np.mean(len_list))
        max_len = int(np.max(len_list))
        num = len(len_list)
        ids_num = len(set(v))
        vocab_size = min(ids_num + 5000, 10 * (ids_num+10))
        if (vocab_size < min_hash_buckets):
            vocab_size = min_hash_buckets
        if (vocab_size > max_hash_buckets):
            vocab_size = max_hash_buckets
        print(f' varlen_str_sparse_feat : {k} ')
        print('mean_len', mean_len)
        print('max_len', max_len)
        print('num', num)
        print('ids_num', ids_num)
        print('vocab_size', vocab_size)

        mapping_dict_name = f'{k}.dict'
        mapping_dict_path = os.path.join(id_map_dict_path, mapping_dict_name)
        feat_num_statis_dict_path = os.path.join(id_map_dict_path, f'num_static.{k}.dict')

        feat_counts = Counter(v)
        save_feat_id_dict(mapping_dict_path, feat_num_statis_dict_path, k, feat_counts)

        varlen_sparse_features.append( {'name' : k,
                'value_type' : 'str',
                'ids_num' : ids_num,
                'vocab_size' : vocab_size,
                "mapping_type" : sparse_feat_mapping_type,
                'mapping_dict_name' : mapping_dict_name,
                'shared_embedding_name' : '',
                'max_len' : min(seq_feat_max_len, max_len),
                'pooling_type' : seq_feat_pooling_type
                } )

    with open(os.path.join(args.output_path, 'feature_config.json'), 'w') as f:
        feat_config = {"dense_features" : dense_features, "sparse_features" : sparse_features, "varlen_sparse_features" : varlen_sparse_features}
        feat_config_str = json.dumps(feat_config, indent=4) # sort_keys=True, 
        f.write(feat_config_str)
        


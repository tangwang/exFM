import sys
import logging
import traceback
import cgitb
import os
from optparse import OptionParser
import argparse
import conf
import numpy as np
import pickle

dump_filename = 'feadump/feamap.pkl'

min_freq_of_sparse_fea = 10
skip_head_tails_of_dense_fea = 10

feamap_dict = {}

def dump():
    global feamap_dict
    global dump_filename
    fea_statis = {}
    for line in sys.stdin:
        segs = line.rstrip().split('\t')[1:]
        for kv in segs:
            k, v = kv.split(':')
            if k == 'bid': # 特殊处理
                v = str(int(round(float(v))))
    
            if k in conf.varlenSparseFeaList.keys() or k in conf.sparseFeaList:
                fea_statis.setdefault(k, {})
                for iv in set(v.split(',')):
                    fea_statis[k].setdefault(iv, 0)
                    fea_statis[k][iv] += 1
            elif k in conf.denseFeaList:
                v =  float(v)
                # min max num sum
                fea_statis.setdefault(k, [])
                fea_statis[k].append(v)
    
    # 保存
    for feaName in conf.sparseFeaList:
        if not feaName in fea_statis: continue
        feaStatis = fea_statis[feaName]
        value_map_dict = {}
        with open(f'feadump/sparse.{feaName}', 'w') as f:
            print('mappedID', 'origID', 'count', sep='\t', file=f)
            for idx, (k,v) in enumerate(sorted(feaStatis.items(), key=lambda x:x[1], reverse=True)):
                
                mappedID = 1 if v < min_freq_of_sparse_fea else idx + 2   # 特征从2开始编号，0留给PAD，1留给unknown
                if mappedID != 1:
                    value_map_dict[k] = mappedID
                print(mappedID, k, v, sep='\t', file=f)
        feamap_dict[feaName] = (conf.fea_type_sparce, value_map_dict)
    
    for feaName in conf.varlenSparseFeaList.keys():
        if not feaName in fea_statis: continue
        feaStatis = fea_statis[feaName]
        value_map_dict = {}
        with open(f'feadump/seq.{feaName}', 'w') as f:
            print('mappedID', 'origID', 'count', sep='\t', file=f)
            for idx, (k,v) in enumerate(sorted(feaStatis.items(), key=lambda x:x[1], reverse=True)):
                
                mappedID = 1 if v < min_freq_of_sparse_fea else idx + 2   # 特征从2开始编号，0留给PAD，1留给unknown
                if mappedID != 1:
                    value_map_dict[k] = mappedID
                print(mappedID, k, v, sep='\t', file=f)
        feamap_dict[feaName] = (conf.fea_type_varlen_sparce, value_map_dict)
    
    
    for feaName in conf.denseFeaList:
        if not feaName in fea_statis: continue
        feaStatis = fea_statis[feaName]
        with open(f'feadump/dense.{feaName}', 'w') as f:
            print('min, max, num, 10 buckets by freq, 50 buckets for freq',  file=f)
            feaStatis.sort()
            feaStatis = feaStatis[skip_head_tails_of_dense_fea : -skip_head_tails_of_dense_fea]
            # min, max, 10 buckets by freq, 50 buckets for freq
            #print(feaName, len(feaStatis), len(fea_statis[feaName]))
            statis_info = (feaStatis[0], feaStatis[-1], len(fea_statis[feaName]), [np.percentile(feaStatis, x) for x in range(0,100,10)],  [np.percentile(feaStatis, x) for x in range(0,100,2)])
            print(statis_info,  file=f)
    
        feamap_dict[feaName] = (conf.fea_type_dense, statis_info)
    
    pickle.dump(feamap_dict, open(dump_filename,'wb'), pickle.HIGHEST_PROTOCOL)

def load():
    global feamap_dict
    global dump_filename
    feamap_dict = pickle.load(open(dump_filename, 'rb'))
    #print('load finished')
    #print(f'len {len(feamap_dict)}')
    #for k, v in feamap_dict.items():
    #    print(k, v[0], len(v[1]))

def map_fea(k, v):
    global feamap_dict
    if not k in feamap_dict : return ''

    
    (fea_type, fea_map_dict) = feamap_dict[k]
    # 特殊处理：这些字段，不做映射：
    if fea_type == conf.fea_type_varlen_sparce or fea_type == conf.fea_type_sparce:
        return k + ':' + v

    if fea_type == conf.fea_type_varlen_sparce:
        # unknown 为 1， 0留给PAD
        mapped_list = [str(fea_map_dict.get(x, 1)) for x in v.split(',')]
        return k + ':' + ','.join(mapped_list)
    elif fea_type == conf.fea_type_sparce:
        return k + ':' + str(fea_map_dict.get(v, 1))
    elif fea_type == conf.fea_type_dense:
        (minV, maxV, num, buckets_10, buckets_50) = fea_map_dict
        if (minV == maxV):
            return ''
        v = float(v)
        if v < minV : v = minV
        if v > maxV : v = maxV
        wide_buckets_10 =  round(10 * (v - minV) / (maxV - minV))
        wide_buckets_50 =  round(50 * (v - minV) / (maxV - minV))
        freq_buckets_10 =  np.searchsorted(buckets_10, v)
        freq_buckets_50 =  np.searchsorted(buckets_50, v)
        mapped_list = (wide_buckets_10, wide_buckets_50, freq_buckets_10, freq_buckets_50)
        return k + ':' + ','.join([str(x) for x in mapped_list])

def map_line(line):
    segs = line.split('\t')
    label = segs[0]
    x = segs[1:]
    for i in range(1, len(segs)):
        k, v = segs[i].split(':')
        mapped_fea = map_fea(k, v)
        segs[i] = mapped_fea
    print(' '.join(segs))


def to_np_data(args):

    denseFeaMap = {}
    for fea_name in conf.denseFeaList:
        denseFeaMap[f' {fea_name}:'] = []
    sparseFeaMap = {}
    for fea_name in conf.sparseFeaList:
        sparseFeaMap[f' {fea_name}:'] = []
    seqFeaMap = {}
    for fea_name, max_len in conf.varlenSparseFeaList.items():
        seqFeaMap[f' {fea_name}:'] = (max_len, [], [])

    print(denseFeaMap)
    label_list = []
    for line in sys.stdin:
        line = line.rstrip()

        label = int(line[0])
        label_list.append(label)

        for fea_name, l in denseFeaMap.items():
            beg = line.find(fea_name)
            if beg != -1:
                beg += len(fea_name)
                fea_v =  line[beg:line.find(' ', beg)]
                fea_value =  (0.0, 0.0, 0.0, 0.0) if not fea_v else [int(round(float(x))) for x in fea_v.split(',')]
            else:
                fea_value = (0.0, 0.0, 0.0, 0.0)
            l.append(fea_value)
            
        for fea_name, l in sparseFeaMap.items():
            beg = line.find(fea_name)
            if beg != -1:
                beg += len(fea_name)
                fea_v =  line[beg:line.find(' ', beg)]
                if ',' in fea_v:  # 特殊处理，这里不应该有multihot，特殊兼容一下
                    fea_v = fea_v[:fea_v.find(',')]
                fea_value =  0 if not fea_v else int(round(float(fea_v)))
            else:
                fea_value = 0
            l.append(fea_value)
            
        for fea_name, (max_len, l_v, l_len) in seqFeaMap.items():
            beg = line.find(fea_name)
            fea_value = [0] * max_len
            seq_len = 0
            if beg != -1:
                beg += len(fea_name)
                fea_v =  line[beg:line.find(' ', beg)]
                if fea_v:
                    for idx, x in enumerate(fea_v.split(',')[:max_len]):
                        seq_len += 1
                        if x:
                            fea_value[idx] = int(round(float(x)))
            
            l_v.append(fea_value)
            l_len.append(seq_len)
 
    if True:
        for fea_name, l in denseFeaMap.items():
            fea_name = fea_name[1:-1]
            np.save(f'{args.output}/{fea_name}', np.array(l))
    
        for fea_name, l in sparseFeaMap.items():
            fea_name = fea_name[1:-1]
            np.save(f'{args.output}/{fea_name}', np.array(l))
    
        for fea_name, (max_len, l_v, l_len) in seqFeaMap.items():
            fea_name = fea_name[1:-1]
            np.save(f'{args.output}/{fea_name}', np.array(l_v))
            np.save(f'{args.output}/{fea_name}_len', np.array(l_len))

        np.save(f'{args.output}/y', np.array(label_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode', '--mode', type=str)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    if args.mode == 'dump':
        dump()
    elif args.mode == 'conv':
        load()
        for line in sys.stdin:
            map_line(line.rstrip())
    elif args.mode == 'to_np':
        load()
        to_np_data(args)
    else:
        print('args.mode must be dump or conv!!!', file=sys.stderr)


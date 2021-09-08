/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/common_fea.h"

class DenseFeaConfig : public CommonFeaConfig {
 public:
  real_t min;
  real_t max;
  real_t default_value;
  mutable shared_ptr<ParamContainer> param_container;
  mutable vector<Mutex_t> mutexes;
  Mutex_t * GetMutexByBucketID(int bucket_id) const {
    return &mutexes[bucket_id];
  }

  // 配置的等频分桶桶宽
  vector<int> samewide_bucket_nums;
  // 配置的分桶
  vector<vector<real_t>> bucket_splits;

  // 以下3个vector，长度一致，按位置一一对应
  vector<real_t> all_splits;                        // 分隔值
  vector<vector<feaid_t>> fea_ids_of_each_buckets;  // 分隔值对应的onehot ID列表
  vector<vector<FTRLParamUnit *>>
      fea_params_of_each_buckets;  // 分隔值对应的onehot ID列表 所对应的参数位置

  const vector<feaid_t> &get_fea_ids(real_t x) const {
    if (x == default_value) {
      return fea_ids_of_each_buckets[fea_ids_of_each_buckets.size() - 1];
    }
    int bucket_id = lower_bound(all_splits.begin(), all_splits.end(), x) -
                    all_splits.begin();
    // TODO check
    if (bucket_id == (int)all_splits.size()) --bucket_id;
    /* gdb debug
     p fea_params_of_each_buckets[bucket_id]
     拿到param地址后：
     p (*(FTRLParamUnit *)0x6c8138)
     p (*(FTRLParamUnit *)0x6c8138).buff@24
     */
    return fea_ids_of_each_buckets[bucket_id];
  }

  int get_fea_bucket_id(real_t x) const {
    assert(x != default_value);
    int bucket_id = lower_bound(all_splits.begin(), all_splits.end(), x) -
                    all_splits.begin();

    // TODO是否要加这个
    if (bucket_id == (int)all_splits.size()) --bucket_id;

    return bucket_id;
  }

  int initParams();

  void dump() const {
    cout << "------------------------------------- \n";
    cout << " DenseFeaConfig name <" << name << ">\n";
    cout << " bucket_splits <\n";
    utils::print2dArray(bucket_splits);
    cout << ">\n samewide_bucket_nums <\n";
    utils::printArray(samewide_bucket_nums);
    cout << ">\n all_splits <\n ";
    utils::printArray(all_splits);
    cout << ">\n fea_ids_of_each_buckets <\n ";
    utils::print2dArray(fea_ids_of_each_buckets);

    cout << ">\n min <" << min << "> max <" << max << ">\n";
    cout << " default_value <" << default_value << ">\n";
  }

  DenseFeaConfig();
  ~DenseFeaConfig();
};

void to_json(json &j, const DenseFeaConfig &p);
void from_json(const json &j, DenseFeaConfig &p);

class DenseFeaContext : public CommonFeaContext {
 public:
  real_t orig_x;
  const vector<FTRLParamUnit *> *fea_params;

  const DenseFeaConfig &cfg_;

  int feedRawData(const char *line, vector<ParamContext> & forward_params, vector<ParamContext> & backward_params);

  bool valid() const {
    // TODO 暂时只支持离散特征
    return orig_x != cfg_.default_value && !cfg_.all_splits.empty();
  }

  void forward(vector<ParamContext> &forward_params);

  void backward();

  DenseFeaContext(const DenseFeaConfig &cfg);
  ~DenseFeaContext();
};

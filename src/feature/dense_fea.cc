/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/dense_fea.h"

DenseFeaConfig::DenseFeaConfig() {}

DenseFeaConfig::~DenseFeaConfig() {}

int DenseFeaConfig::initParams() {
  const int onehot_fea_dimension =
      samewide_bucket_nums.size() + bucket_splits.size();
  if (onehot_fea_dimension == 0) return 0;

  ftrl_param = std::make_shared<FtrlParamContainer>(onehot_fea_dimension);

  vector<pair<real_t, vector<feaid_t>>> all_split_position_and_mapping_ids;
  int onehot_dimension = 0;
  int onehot_id = 0;
  for (int bucket_num : samewide_bucket_nums) {
    vector<feaid_t> onehot_values(onehot_fea_dimension);
    real_t wide = (max - min) / bucket_num;
    // TODO check边界
    for (int bucket_id = 0; bucket_id < bucket_num; bucket_id++) {
      onehot_values[onehot_dimension] = onehot_id++;
      all_split_position_and_mapping_ids.push_back(
          std::make_pair(min + bucket_id * wide, onehot_values));
    }
    onehot_dimension++;
  }

  for (auto buckets : bucket_splits) {
    vector<feaid_t> onehot_values(onehot_fea_dimension);
    // 因为离散化时是取lower_bound的idx，所以min值也放进去
    onehot_values[onehot_dimension] = onehot_id++;
    all_split_position_and_mapping_ids.push_back(
        std::make_pair(min, onehot_values));
    // 配置指定的各个分隔位置
    for (auto split_value : buckets) {
      onehot_values[onehot_dimension] = onehot_id++;
      all_split_position_and_mapping_ids.push_back(
          std::make_pair(split_value, onehot_values));
    }
    onehot_dimension++;
  }

  sort(all_split_position_and_mapping_ids.begin(),
       all_split_position_and_mapping_ids.end(),
       utils::judgeByPairFirst<real_t, vector<feaid_t>>);

  for (int split_idx = 1; split_idx < all_split_position_and_mapping_ids.size();
       split_idx++) {
    // 补全各个维度的fea_id
    for (int dimension = 0; dimension < onehot_fea_dimension; dimension++) {
      feaid_t &this_id =
          all_split_position_and_mapping_ids[split_idx].second[dimension];
      feaid_t last_id =
          all_split_position_and_mapping_ids[split_idx - 1].second[dimension];

      this_id = std::max(this_id, last_id);
    }

    // 记录映射关系
    // 分隔值
    all_splits.push_back(all_split_position_and_mapping_ids[split_idx].first);
    // 分桶内的各维度的fea_id
    fea_ids_of_each_buckets.push_back(
        all_split_position_and_mapping_ids[split_idx].second);
  }

  // 提前取出参数位置
  fea_params_of_each_buckets.resize(fea_ids_of_each_buckets.size());
  for (size_t i = 0; i < fea_ids_of_each_buckets.size(); i++) {
    auto &i_value = fea_ids_of_each_buckets[i];
    fea_params_of_each_buckets[i].resize(i_value.size());
    for (size_t j = 0; j < i_value.size(); j++) {
      fea_params_of_each_buckets[i][j] = ftrl_param->get(i_value[j]);
    }
  }

  // initail mutexes
  mutexes.resize(fea_params_of_each_buckets.size());

  return 0;
}

void to_json(json &j, const DenseFeaConfig &p) {
  j = json{{"name", p.name},
           {"min", p.min},
           {"max", p.max},
           {"default_value", p.default_value},
           {"samewide_bucket_nums", p.samewide_bucket_nums},
           {"bucket_splits", p.bucket_splits}};
}

void from_json(const json &j, DenseFeaConfig &p) {
  j.at("name").get_to(p.name);
  j.at("min").get_to(p.min);
  j.at("max").get_to(p.max);
  j.at("default_value").get_to(p.default_value);
  j.at("samewide_bucket_nums").get_to(p.samewide_bucket_nums);
  j.at("bucket_splits").get_to(p.bucket_splits);
}

DenseFeaContext::DenseFeaContext(const DenseFeaConfig &cfg) : cfg_(cfg) {}

DenseFeaContext::~DenseFeaContext() {}

int DenseFeaContext::feedSample(const char *line,
                                 vector<ParamContext> &forward_params,
                                 vector<ParamContext> &backward_params) {
  cfg_.parseReal(line, orig_x, cfg_.default_value);

  if (!valid()) {
    return -1;
  }
  int bucket_id = cfg_.get_fea_bucket_id(orig_x);
  fea_params = &cfg_.fea_params_of_each_buckets[bucket_id];

  FtrlParamUnit *forward_param = forward_param_container->get();
  forward_param->clear_weights();
  for (auto fea_param : *fea_params) {
    Mutex_t *param_mutex = cfg_.GetMutexByBucketID(bucket_id);
    backward_params.push_back(ParamContext(fea_param, param_mutex));

    param_mutex->lock();
    fea_param->calc_param();
    forward_param->plus_weights(*fea_param);
    param_mutex->unlock();
  }

  forward_params.push_back(ParamContext(forward_param, NULL));
  return 0;
}

void DenseFeaContext::forward(vector<ParamContext> &forward_params) {}

void DenseFeaContext::backward() {
  FtrlParamUnit *p = backward_param_container->get();

  for (FtrlParamUnit *fea_param : *fea_params) {
    fea_param->plus_params(*p);
  }
}

/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/dense_feat.h"
#include "solver/solver_factory.h"

DenseFeatConfig::DenseFeatConfig() {
  default_value = 0.0;
}

DenseFeatConfig::~DenseFeatConfig() {}

bool DenseFeatConfig::initParams(unordered_map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) {
  const feat_id_t onehot_feat_dimension =
      sparse_by_wide_bins_numbs.size() + sparse_by_splits.size();
  if (onehot_feat_dimension == 0) {
    // 对dense特征的使用，暂时只支持稀疏化，不支持使用原始值
    cerr << " no sparcity method for dense feature " << name << endl;
    return false;
  }
  
  vector<pair<real_t, vector<feat_id_t>>> all_split_position_and_mapping_ids;
  feat_id_t onehot_dimension = 0;
  feat_id_t onehot_id = 0;
  for (int bucket_num : sparse_by_wide_bins_numbs) {
    vector<feat_id_t> onehot_values(onehot_feat_dimension);
    real_t wide = (max - min) / bucket_num;
    // TODO check边界
    for (int bucket_id = 0; bucket_id < bucket_num; bucket_id++) {
      onehot_values[onehot_dimension] = onehot_id++;
      all_split_position_and_mapping_ids.push_back(
          std::make_pair(min + bucket_id * wide, onehot_values));
    }
    onehot_dimension++;
  }

  for (auto buckets : sparse_by_splits) {
    vector<feat_id_t> onehot_values(onehot_feat_dimension);
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
       utils::judgeByPairFirst<real_t, vector<feat_id_t>>);

  for (size_t split_idx = 1; split_idx < all_split_position_and_mapping_ids.size();
       split_idx++) {
    // 补全各个维度的feat_id
    for (size_t dimension = 0; dimension < onehot_feat_dimension; dimension++) {
      feat_id_t &this_id =
          all_split_position_and_mapping_ids[split_idx].second[dimension];
      feat_id_t last_id =
          all_split_position_and_mapping_ids[split_idx - 1].second[dimension];

      this_id = std::max(this_id, last_id);
    }

    // 记录映射关系
    // 分隔值
    all_splits.push_back(all_split_position_and_mapping_ids[split_idx].first);
    // 分桶内的各维度的feat_id
     feat_ids_of_each_buckets.push_back(
        all_split_position_and_mapping_ids[split_idx].second);
  }

  param_container = creatParamContainer(onehot_feat_dimension, (feat_id_t)feat_ids_of_each_buckets.size());
  loadModel();

  // 提前取出参数位置
   feat_params_of_each_buckets.resize(feat_ids_of_each_buckets.size());
  for (size_t i = 0; i <  feat_ids_of_each_buckets.size(); i++) {
    auto &i_value =  feat_ids_of_each_buckets[i];
     feat_params_of_each_buckets[i].resize(i_value.size());
    for (size_t j = 0; j < i_value.size(); j++) {
       feat_params_of_each_buckets[i][j] = param_container->get(i_value[j]);
    }
  }

  return true;
}

void to_json(json &j, const DenseFeatConfig &p) {
  j = json{{"name", p.name},
           {"min_clip", p.min},
           {"max_clip", p.max},
           {"default_value", p.default_value},
           {"sparse_by_wide_bins_numbs", p.sparse_by_wide_bins_numbs},
           {"sparse_by_splits", p.sparse_by_splits}};
}

void from_json(const json &j, DenseFeatConfig &p) {
  if (j.find("name") == j.end()) {
    throw "feature config err : no attr \"name\" in dense feature.";
  }
  j.at("name").get_to(p.name);

  if (j.find("min_clip") == j.end()) {
    throw "feature config err : no attr \"min_clip\" in dense feature.";
  }

  j.at("max_clip").get_to(p.min);
  if (j.find("max_clip") == j.end()) {
    throw "feature config err : no attr \"max_clip\" in dense feature.";
  }

  j.at("max_clip").get_to(p.max);
  if (j.find("default_value") != j.end())      j.at("default_value").get_to(p.default_value);
  if (j.find("sparse_by_wide_bins_numbs") != j.end())     j.at("sparse_by_wide_bins_numbs").get_to(p.sparse_by_wide_bins_numbs);
  if (j.find("sparse_by_splits") != j.end())   j.at("sparse_by_splits").get_to(p.sparse_by_splits);
}

DenseFeatContext::DenseFeatContext(const DenseFeatConfig &cfg) : cfg_(cfg) {
  feat_cfg = &cfg_;
}

DenseFeatContext::~DenseFeatContext() {}

int DenseFeatContext::feedSample(const char *feat_str, size_t feat_str_len, FmLayerNode & fm_node) {
  orig_x = atof(feat_str);

  if (!valid()) { // TODO remove
    fm_node.forward.clear();
    fm_node.backward_nodes.clear();
    return -1;
  }
  int bucket_id = cfg_.getFeaBucketId(orig_x);
   feat_params = &cfg_.feat_params_of_each_buckets[bucket_id];

  DEBUG_OUT << "feedSample " << cfg_.name << " orig_x " << orig_x << " bucket_id " << bucket_id << endl;

  fm_node.forward.clear();
  fm_node.backward_nodes.clear();

  for (auto  feat_param : *feat_params) {
    Mutex_t *param_mutex = cfg_.param_container->GetMutexByFeatID(bucket_id);
    fm_node.backward_nodes.emplace_back(feat_param, param_mutex, 1.0, 1.0);
    param_mutex->lock();
    fm_node.forward += *feat_param;
    param_mutex->unlock();
  }

  return 0;
}


/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/varlen_sparse_feat.h"
#include "solver/solver_factory.h"

VarlenSparseFeatConfig::VarlenSparseFeatConfig() {}

VarlenSparseFeatConfig::~VarlenSparseFeatConfig() {}

VarlenSparseFeatContext::VarlenSparseFeatContext(const VarlenSparseFeatConfig &cfg)
    : cfg_(cfg) {
  feat_cfg = &cfg_;
   feat_ids.reserve(cfg_.max_len);
}

VarlenSparseFeatContext::~VarlenSparseFeatContext() {}

bool VarlenSparseFeatConfig::initParams(unordered_map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) {
  bool ret = sparse_cfg.initParams(shared_param_container_map);
  // 保存model会用到。直接引用内部sparse_cfg的param_container
  param_container = sparse_cfg.param_container;

  return ret;
}

void to_json(json &j, const VarlenSparseFeatConfig &p) {
  j = json{{"name", p.name},
           {"max_id", p.sparse_cfg.max_id},
           {"mapping_dict_name", p.sparse_cfg.mapping_dict_name},
           {"default_id", p.sparse_cfg.default_id},
           {"unknown_id", p.sparse_cfg.unknown_id},
           {"max_len", p.max_len},
           {"shared_embedding_name", p.sparse_cfg.shared_embedding_name},
           {"pooling_type", (int)p.pooling_type_id}
           };
}

void from_json(const json &j, VarlenSparseFeatConfig &p) {
  if (!j.contains("name")) {
    throw "feature config err : no attr \"name\" in varlen_sparse feature.";
  }
  if (!j.contains("max_len")) {
    throw "feature config err : no attr \"max_len\" in varlen_sparse feature.";
  }
  j.at("name").get_to(p.name);
  j.at("max_len").get_to(p.max_len);

  string pooling_type = "sum";
  if (j.contains("pooling_type")) {
    j.at("pooling_type").get_to(pooling_type);
  }
  if (pooling_type == "sum") {
    p.pooling_type_id = VarlenSparseFeatConfig::SeqPoolTypeSUM;
  } else if (pooling_type == "avg") {
    p.pooling_type_id = VarlenSparseFeatConfig::SeqPoolTypeAVG;
  } else {
    std::cerr << "Not supported.  use sum pooling." << endl;
    p.pooling_type_id = VarlenSparseFeatConfig::SeqPoolTypeSUM;
    throw "feature config err : Not supported pooling type. only support sum/avg";
  }

  from_json(j, p.sparse_cfg);
}

int VarlenSparseFeatContext::feedSample(const char *feat_str, size_t feat_str_len, FmLayerNode & fm_node) {
  // parse feat ids
   feat_ids.clear();
  const char *featid_beg = feat_str;
  const char *p = feat_str;
  for (; *p; p++) {
    if (*p == train_opt.feat_value_list_seperator) {
      int item_len = std::min((int)(p - featid_beg), (int)sizeof(feat_id_buff)-1);
      if (item_len > 0) {
        memcpy(feat_id_buff, featid_beg, item_len);
        feat_id_buff[item_len] = '\0';
        feat_id_t mapped_id = cfg_.sparse_cfg.featMapping(feat_id_buff, item_len);
        if (likely(mapped_id != cfg_.sparse_cfg.default_id))
          feat_ids.push_back(mapped_id);
        if (feat_ids.size() == cfg_.max_len) break;
      }
      featid_beg = p + 1;
    }
  }
  int item_len = std::min((int)(p - featid_beg), (int)sizeof(feat_id_buff)-1);
  if (item_len > 0 &&  feat_ids.size() != cfg_.max_len) {
    memcpy(feat_id_buff, featid_beg, item_len);
    feat_id_buff[item_len] = '\0';
    feat_id_t mapped_id = cfg_.sparse_cfg.featMapping(feat_id_buff, item_len);
    if (likely(mapped_id != cfg_.sparse_cfg.default_id))
      feat_ids.push_back(mapped_id);
  }

  fm_node.forward.clear();
  fm_node.backward_nodes.clear();

  if (!valid()) {
    return -1;
  }

  DEBUG_OUT << "feedSample " << cfg_.name << " feat_str " << feat_str << "  feat_ids " <<  feat_ids << endl;

  real_t grad_from_fm_node = 1.0;
  if (cfg_.pooling_type_id == VarlenSparseFeatConfig::SeqPoolTypeAVG) {
    grad_from_fm_node = 1.0 /  feat_ids.size();
  }

  for (auto id :  feat_ids) {
    FMParamUnit *feat_param = cfg_.sparse_cfg.param_container->get(id);
    ParamMutex_t *param_mutex = cfg_.sparse_cfg.param_container->GetMutexByFeatID(id);

    param_mutex->readLock();
    fm_node.forward += *feat_param;
    param_mutex->unlock();
    
    fm_node.backward_nodes.emplace_back(feat_param, param_mutex, 1.0, grad_from_fm_node);
  }

  return 0;
}

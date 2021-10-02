/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/varlen_sparse_feat.h"
#include "solver/solver_factory.h"

VarlenSparseFeatConfig::VarlenSparseFeatConfig() { pooling_type = "sum"; }

VarlenSparseFeatConfig::~VarlenSparseFeatConfig() {}

VarlenSparseFeatContext::VarlenSparseFeatContext(const VarlenSparseFeatConfig &cfg)
    : cfg_(cfg) {
  fea_ids.reserve(cfg_.max_len);
}

VarlenSparseFeatContext::~VarlenSparseFeatContext() {}

bool VarlenSparseFeatConfig::initParams(map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) {
  if (pooling_type == "sum") {
    pooling_type_id = SeqPoolTypeSUM;
  } else if (pooling_type == "avg") {
    pooling_type_id = SeqPoolTypeAVG;
  } else {
    std::cerr << "Not supported.  use sum pooling." << endl;
    pooling_type_id = SeqPoolTypeSUM;
  }

  bool ret = sparse_cfg.initParams(shared_param_container_map);
  // 保存model会用到。直接引用内部sparse_cfg的param_container
  param_container = sparse_cfg.param_container;

  return ret;
}

void to_json(json &j, const VarlenSparseFeatConfig &p) {
  j = json{{"name", p.name},
           {"vocab_size", p.sparse_cfg.vocab_size},
           {"mapping_dict_name", p.sparse_cfg.mapping_dict_name},
           {"default_id", p.sparse_cfg.default_id},
           {"unknown_id", p.sparse_cfg.unknown_id},
           {"max_len", p.max_len},
           {"shared_embedding_name", p.sparse_cfg.shared_embedding_name},
           {"pooling_type", p.pooling_type}};
}

void from_json(const json &j, VarlenSparseFeatConfig &p) {
  if (j.find("name") == j.end()) {
    throw "feature config err : no attr \"name\" in varlen_sparse feature.";
  }
  if (j.find("max_len") == j.end()) {
    throw "feature config err : no attr \"max_len\" in varlen_sparse feature.";
  }
  j.at("name").get_to(p.name);
  j.at("max_len").get_to(p.max_len);
  
  from_json(j, p.sparse_cfg);
}

int VarlenSparseFeatContext::feedSample(const char *line,
                                        vector<ParamContext> &forward_params,
                                        vector<ParamContext> &backward_params) {
  cfg_.parseStrList(line, orig_fea_ids);
  if (orig_fea_ids.size() > cfg_.max_len) {
    orig_fea_ids.resize(cfg_.max_len);
  }

  fea_params.clear();
  fea_ids.clear();
  for (auto orig_fea_id : orig_fea_ids) {
    feaid_t mapped_id = cfg_.sparse_cfg.featMapping(orig_fea_id);
    fea_ids.push_back(mapped_id);
  }
  if (!valid()) {
    return -1;
  }

  DEBUG_OUT << "feedSample " << cfg_.name << " orig_fea_ids " << orig_fea_ids << " fea_ids " << fea_ids << endl;

  FMParamUnit *forward_param = forward_param_container->get();
  forward_param->clear();

  for (auto id : fea_ids) {
    FMParamUnit *fea_param = cfg_.sparse_cfg.param_container->get(id);
    Mutex_t *param_mutex = cfg_.sparse_cfg.param_container->GetMutexByFeaID(id);

    param_mutex->lock();
    *forward_param += *fea_param;
    param_mutex->unlock();
    
    fea_params.push_back(fea_param);
    backward_params.push_back(ParamContext((ParamContainerInterface*)cfg_.sparse_cfg.param_container.get(), fea_param, param_mutex, 1.0));
  }

  forward_params.push_back(ParamContext((ParamContainerInterface*)cfg_.sparse_cfg.param_container.get(), forward_param, NULL, 1.0));

  return 0;
}

void VarlenSparseFeatContext::forward(vector<ParamContext> &forward_params) {}

void VarlenSparseFeatContext::backward() {
}

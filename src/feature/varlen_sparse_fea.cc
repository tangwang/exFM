/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/varlen_sparse_fea.h"

VarlenSparseFeaConfig::VarlenSparseFeaConfig() {}

VarlenSparseFeaConfig::~VarlenSparseFeaConfig() {}

VarlenSparseFeaContext::VarlenSparseFeaContext(const VarlenSparseFeaConfig &cfg)
    : cfg_(cfg) {
  fea_ids.reserve(cfg_.max_len);
}

VarlenSparseFeaContext::~VarlenSparseFeaContext() {}

int VarlenSparseFeaConfig::initParams() {
  if (pooling_type == "sum") {
    pooling_type_id = SeqPoolTypeSUM;
  } else if (pooling_type == "avg") {
    pooling_type_id = SeqPoolTypeAVG;
  } else {
    fprintf(stderr, "Not supported.  use sum pooling.");
    pooling_type_id = SeqPoolTypeSUM;
  }

  sparse_cfg.initParams();

  return 0;
}

void to_json(json &j, const VarlenSparseFeaConfig &p) {
  j = json{{"name", p.name},
           {"max_id", p.sparse_cfg.max_id},
           {"vocab_size", p.sparse_cfg.vocab_size},
           {"id_mapping_dict_path", p.sparse_cfg.id_mapping_dict_path},
           {"use_id_mapping", p.sparse_cfg.use_id_mapping},
           {"use_hash", p.sparse_cfg.use_hash},
           {"default_value", p.sparse_cfg.default_value},
           {"max_len", p.max_len},
           {"pooling_type", p.pooling_type}};
}

void from_json(const json &j, VarlenSparseFeaConfig &p) {
  j.at("name").get_to(p.name);
  j.at("name").get_to(p.sparse_cfg.name);
  j.at("max_id").get_to(p.sparse_cfg.max_id);
  j.at("vocab_size").get_to(p.sparse_cfg.vocab_size);
  j.at("id_mapping_dict_path").get_to(p.sparse_cfg.id_mapping_dict_path);
  j.at("use_hash").get_to(p.sparse_cfg.use_hash);
  j.at("use_id_mapping").get_to(p.sparse_cfg.use_id_mapping);
  j.at("default_value").get_to(p.sparse_cfg.default_value);
  j.at("max_len").get_to(p.max_len);
  j.at("pooling_type").get_to(p.pooling_type);
}

int VarlenSparseFeaContext::feedSample(const char *line,
                                        vector<ParamContext> &forward_params,
                                        vector<ParamContext> &backward_params) {
  cfg_.parseFeaIdList(line, orig_fea_ids);
  if (orig_fea_ids.size() > cfg_.max_len) {
    orig_fea_ids.resize(cfg_.max_len);
  }

  fea_params.clear();
  if (cfg_.sparse_cfg.use_id_mapping == 0) {
    fea_ids = orig_fea_ids;
  } else {
    fea_ids.clear();
    for (auto orig_fea_id : orig_fea_ids) {
      feaid_t mapped_id = cfg_.sparse_cfg.fea_id_mapping.get(orig_fea_id);
      fea_ids.push_back(mapped_id);
    }
  }
  if (!valid()) {
    return -1;
  }

  FtrlParamUnit *forward_param = forward_param_container->get();
  FtrlParamUnit *locak_buff_param = local_buff_container->get();
  forward_param->clear_weights();
  for (auto id : fea_ids) {
    FtrlParamUnit *fea_param = cfg_.sparse_cfg.ftrl_param->get(id);
    Mutex_t *param_mutex = cfg_.sparse_cfg.GetMutexByFeaID(id);

    param_mutex->lock();
    *locak_buff_param = *fea_param;
    param_mutex->unlock();

    locak_buff_param->calc_param();
    forward_param->plus_weights(*locak_buff_param);

    fea_params.push_back(fea_param);
    backward_params.push_back(ParamContext(fea_param, param_mutex));
  }

  forward_params.push_back(ParamContext(forward_param, NULL));

  return 0;
}

void VarlenSparseFeaContext::forward(vector<ParamContext> &forward_params) {}

void VarlenSparseFeaContext::backward() {
  // TODO这个是错误的，  废弃
  FtrlParamUnit *p = backward_param_container->get();

  for (FtrlParamUnit *fea_param : fea_params) {
    fea_param->plus_params(*p);
  }
}

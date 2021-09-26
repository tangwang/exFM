/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/sparse_fea.h"
#include "solver/solver_factory.h"

SparseFeaConfig::SparseFeaConfig() {
  default_value = -1;
  use_id_mapping = 0;
  use_hash = false;
}

SparseFeaConfig::~SparseFeaConfig() {}

int SparseFeaConfig::initParams(map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) {
  // TODO 暂时都不用词典映射
  if (train_opt.disable_feaid_mapping) {
    use_id_mapping = 0;
    vocab_size = max_id + 2;
  }

  // initail mutexes
  feaid_t mutex_nums = vocab_size;
  if (mutex_nums > 10000) {
    mutex_nums = std::max(10000, (int)std::pow((float)mutex_nums, 0.8));
    mutex_nums = std::min(mutex_nums, (feaid_t)1000000);
  }

  bool embedding_existed = false;
  if (!shared_embedding_name.empty()) {
    auto iter = shared_param_container_map.find(shared_embedding_name);
    if (iter != shared_param_container_map.end()) {
      param_container = iter->second;
      embedding_existed = true;
    }
  }
  if (!embedding_existed) {
    param_container = creatParamContainer(vocab_size, mutex_nums);
    if (!shared_embedding_name.empty()) shared_param_container_map[shared_embedding_name] = param_container;
    // 共享embedding的feature，param_container的创建者负责warmup，以保证只warm_start()一次
    loadModel();
  }


  if (use_id_mapping != 0 && !id_mapping_dict_path.empty()) {
    // assert(access(id_mapping_dict_path.c_str(), F_OK) != -1 &&
    // ("id_mapping_dict_path doesn't exist: " + id_mapping_dict_path));
    assert(access(id_mapping_dict_path.c_str(), F_OK) != -1);
    if (fea_id_mapping.create(id_mapping_dict_path,
                              train_opt.fea_id_mapping_dict_seperator)) {
      std::cout << "load dict <" << id_mapping_dict_path << "> ok, size <"
                << fea_id_mapping.size() << ">" << std::endl;
    } else {
      std::cerr << "load dict <" << id_mapping_dict_path << "> failed!!!"
                << std::endl;
    }
  }

  return 0;
}

void to_json(json &j, const SparseFeaConfig &p) {
  j = json{{"name", p.name},
           {"vocab_size", p.vocab_size},
           {"id_mapping_dict_path", p.id_mapping_dict_path},
           {"use_id_mapping", p.use_id_mapping},
           {"max_id", p.max_id},
           {"use_hash", p.use_hash},
           {"shared_embedding_name", p.shared_embedding_name},
           {"default_value", p.default_value}};
}

void from_json(const json &j, SparseFeaConfig &p) {
  j.at("name").get_to(p.name);
  j.at("vocab_size").get_to(p.vocab_size);
  if (j.find("use_id_mapping") != j.end())             j.at("use_id_mapping").get_to(p.use_id_mapping);
  if (j.find("max_id") != j.end())                     j.at("max_id").get_to(p.max_id);
  if (j.find("use_hash") != j.end())                   j.at("use_hash").get_to(p.use_hash);
  if (j.find("id_mapping_dict_path") != j.end())       j.at("id_mapping_dict_path").get_to(p.id_mapping_dict_path);
  
  if (j.find("shared_embedding_name") != j.end())      j.at("shared_embedding_name").get_to(p.shared_embedding_name);
  if (j.find("default_value") != j.end())              j.at("default_value").get_to(p.default_value);
}

SparseFeaContext::SparseFeaContext(const SparseFeaConfig &cfg) : cfg_(cfg) {}

SparseFeaContext::~SparseFeaContext() {}

void SparseFeaContext::forward(vector<ParamContext> &forward_params) {}

int SparseFeaContext::feedSample(const char *line,
                                  vector<ParamContext> &forward_params,
                                  vector<ParamContext> &backward_params) {
  cfg_.parseID(line, orig_fea_id, cfg_.default_value);

  fea_id = cfg_.use_id_mapping == 0 ? orig_fea_id
                                    : cfg_.fea_id_mapping.get(orig_fea_id);
  if (!valid()) {
    return -1;
  }

  DEBUG_OUT << "feedSample " << cfg_.name << " orig_fea_id " << orig_fea_id << " fea_id " << fea_id << endl;

  FMParamUnit *fea_param = cfg_.param_container->get(fea_id);
  Mutex_t *param_mutex = cfg_.param_container->GetMutexByFeaID(fea_id);
  backward_params.push_back(ParamContext((ParamContainerInterface*)cfg_.param_container.get(), fea_param, param_mutex, 1.0));
  FMParamUnit *forward_param = forward_param_container->get();
  param_mutex->lock();
  cfg_.param_container->cpParam(forward_param, fea_param);
  param_mutex->unlock();

  forward_params.push_back(ParamContext((ParamContainerInterface*)cfg_.param_container.get(), forward_param, NULL, 1.0));

  return 0;
}

void SparseFeaContext::backward() {
  // FMParamUnit *p = backward_param_container->get();

  // FMParamUnit *fea_param = cfg_.param_container->get(fea_id);
  // cfg_.sparse_cfg.param_container->addWeightsTo(p, fea_param);
}

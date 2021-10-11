/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
// #include "solver/solver_factory.h"
#include "solver/parammeter_container.h"
#include "nlohmann/json.hpp"
#include "synchronize/mutex_adapter.h"
#include "utils/base.h"
#include "utils/utils.h"
#include <array>

using json = nlohmann::json;

class CommonFeatConfig {
 public:
  string name;
  string identifier;  // fea_seperator + name + kv_seperator
  int identifier_len;

  mutable shared_ptr<ParamContainerInterface> param_container;

  bool loadModel() {
    bool ret = true;
    if (!train_opt.init_model_path.empty()) {
      ret = (0 == param_container->load(train_opt.init_model_path + "/" + name,
                                  train_opt.model_format));
    }
    return ret;
  }

  bool dumpModel() {
    bool ret = true;
    if (!train_opt.model_path.empty()) {
      cout << "dump model for " << name << " ... ";
      if (param_container) {
        if (0 == param_container->dump(train_opt.model_path + "/" + name,
                                    train_opt.model_format)) {
          cout << " ok " << endl;
        } else {
          ret = false;
          cout << " faild " << endl;
        }
      } else {
        cout << " param_container is empty! " << endl;
      }
    }
    return ret;
  }

  virtual bool initParams(map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) = 0;

  bool init(map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) {
    identifier = train_opt.fea_seperator + name + train_opt.fea_kv_seperator;
    identifier_len = identifier.length();

    return initParams(shared_param_container_map);
  }

  bool parseReal(const char *line, real_t &x, real_t default_v) const {
    x = default_v;
    const char *pos = strstr(line, identifier.c_str());
    if (pos != NULL) {
      x = atof(pos + identifier_len);
    }
    return 0;
  }

  bool parseStr(const char *line, string &feaid) const {
    feaid.clear();
    const char *pos = strstr(line, identifier.c_str());
    if (pos != NULL) {
      pos += identifier_len;
      const char *end_pos = strchr(pos, train_opt.fea_seperator);
      feaid = end_pos == NULL ? pos : string(pos, end_pos - pos);
    }
    return true;
  }

  bool parseStrList(const char *line, vector<string> &feaid_list) const {
    feaid_list.clear();
    const char *pos = strstr(line, identifier.c_str());
    if (pos != NULL) {
      pos += identifier_len;

      utils::split_string(pos, train_opt.fea_multivalue_seperator,
                          train_opt.fea_seperator, feaid_list);
    }
    return true;
  }

  bool parseID(const char *line, feaid_t &feaid, feaid_t default_v) const {
    feaid = default_v;
    const char *pos = strstr(line, identifier.c_str());
    if (pos != NULL) {
      feaid = atol(pos + identifier_len);
    }
    return true;
  }

  bool parseFeaIdList(const char *line, vector<feaid_t> &feaid_list) const {
    feaid_list.clear();
    const char *pos = strstr(line, identifier.c_str());
    if (pos != NULL) {
      pos += identifier_len;

      utils::split_string(pos, train_opt.fea_multivalue_seperator,
                          train_opt.fea_seperator, feaid_list);
    }
    return true;
  }
};

class CommonFeatContext {
 public:
  virtual int feedSample(const char *line, FmLayerNode & fm_node) = 0;
  virtual bool valid() const = 0;

  CommonFeatContext() {}

  virtual ~CommonFeatContext() {}
};

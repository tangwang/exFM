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

  virtual bool initParams(unordered_map<string, shared_ptr<ParamContainerInterface>> & shared_param_container_map) = 0;

};

class CommonFeatContext {
 public:
  virtual int feedSample(char *feat_str, FmLayerNode & fm_node) = 0;
  virtual bool valid() const = 0;

  CommonFeatContext() : feat_cfg(NULL) {}

  virtual ~CommonFeatContext() {}

  const CommonFeatConfig * feat_cfg;
};

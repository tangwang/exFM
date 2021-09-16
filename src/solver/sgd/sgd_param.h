/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_interface.h"

class SgdParamUnit {
 public:
  ParamUnitHead head;
  void init_params() {
    head.w = 0.0;
    for (int f = 0; f < train_opt.factor_num; ++f) {
      head.V[f] = utils::gaussian(train_opt.ftrl.init_mean, train_opt.ftrl.init_stdev);
    }
  }
};

class SgdParamContainer : public ParamContainerInterface {
 public:
  SgdParamContainer(feaid_t total_fea_num)
      : ParamContainerInterface(
            total_fea_num,
            sizeof(SgdParamUnit) + train_opt.factor_num * sizeof(real_t)) {
    init_params();
  }

  virtual ~SgdParamContainer() {}

  virtual void init_params() {
    for (feaid_t i = 0; i < fea_num + 1; i++) {
      SgdParamUnit * p = (SgdParamUnit *)get(i);
      p->init_params();
    }
  }

  // virtual void update_param(ParamUnitHead *backward_param, real_t grad) {
  //     real_t lr = train_opt.sgd.step_size;

  //     real_t xi = 1.0;
  //     real_t wg = grad * xi;
  //     backward_param->w -= lr *  wg;

  //     for (int f = 0; f < train_opt.factor_num; ++f) {
  //       real_t &vf = backward_param->V[f];
  //       real_t vgf = wg * (sum[f] - vf * xi);
  //       vf -= lr * vgf;
  //     }
  // }

};

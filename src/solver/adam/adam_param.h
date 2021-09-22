/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class AdamParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit momentum; // 1st moment (the mean) of the gradient
  FMParamUnit variance_m; // 2nd raw moment (the uncentered variance) of the gradient
  real_t beta1power_t;
  real_t beta2power_t;

  AdamParamUnit() {
    fm_param.w = 0.0;
    momentum.w = 0.0;
    variance_m.w = 0.0;
    beta1power_t = 1.0;
    beta2power_t = 1.0;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      momentum.V[f] = 0.0;
      variance_m.V[f] = 0.0;
    }
  }
};


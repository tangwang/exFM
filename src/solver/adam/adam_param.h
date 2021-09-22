/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class AdamParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit avg_grad; // 1st momentum (the mean) of the gradient
  FMParamUnit avg_squared; // 2nd raw momentum (the uncentered variance) of the gradient
  real_t beta1power_t;
  real_t beta2power_t;

  AdamParamUnit() {
    fm_param.w = 0.0;
    avg_grad.w = 0.0;
    avg_squared.w = 0.0;
    beta1power_t = 1.0;
    beta2power_t = 1.0;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      avg_grad.V[f] = 0.0;
      avg_squared.V[f] = 0.0;
    }
  }
};


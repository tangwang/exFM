/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class RmspropParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit avg_squared; // 2nd raw momentum (the uncentered variance) of the gradient

  RmspropParamUnit() {
    fm_param.w = 0.0;
    avg_squared.w = 1e-7;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      avg_squared.V[f] = 1e-7;
    }
  }
};

/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class SgdmParamUnit {
public:
  FMParamUnit fm_wei;
  FMParamUnit momentum;

  SgdmParamUnit() {
    fm_wei.w = 0.0;
    momentum.w = 0.0;
    for (int f = 0; f < DIM; ++f) {
      fm_wei.V[f] = utils::gaussian(train_opt.ftrl.init_mean, train_opt.ftrl.init_stdev);
      momentum.V[f] = 0.0;
    }
  }

};


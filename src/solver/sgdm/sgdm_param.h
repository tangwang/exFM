/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class SgdmParamUnit {
public:
  FMParamUnit fm_param;
  FMParamUnit momentum;

  SgdmParamUnit() {
    fm_param.w = 0.0;
    momentum.w = 0.0;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.ftrl.init_stdev);
      momentum.V[f] = 0.0;
    }
  }

};


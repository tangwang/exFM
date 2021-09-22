/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class AdagradParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit variance_m; // 2nd raw moment (the uncentered variance) of the gradient

  AdagradParamUnit() {
    fm_param.w = 0.0;
    variance_m.w = 0.0;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      variance_m.V[f] = 0.0;
    }
  }
};


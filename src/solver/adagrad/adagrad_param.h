/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class AdagradParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit squared_sum; // 2nd raw momentum (the uncentered variance) of the gradient

  AdagradParamUnit() {
    fm_param.w = 0.0;
    // squared_sum原始论文是初始化为0
    // squared_sum初始化为1e-7相比于初始化为0（原始论文的实现）有提升。初始化为0.1或者1对其他维度的超参（lr, batch_size, l2norm）更为鲁棒，更容易收敛，但是最高AUC不如squared_sum初始化为0的情况。
    squared_sum.w = 1e-7;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(0.0, train_opt.init_stdev);
      squared_sum.V[f] = 1e-7;
    }
  }
};

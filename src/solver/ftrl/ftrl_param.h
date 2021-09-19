/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "solver/parammeter_container.h"

class FtrlParamUnit {
 public:
  FMParamUnit fm_param;
  FMParamUnit n;
  FMParamUnit z;

  FtrlParamUnit() {
    fm_param.w = 0.0;
    n.w = 0.0;
    z.w = 0.0;
    for (int f = 0; f < DIM; ++f) {
      fm_param.V[f] = utils::gaussian(train_opt.ftrl.init_mean, train_opt.ftrl.init_stdev);
      n.V[f] = 0.0;
      z.V[f] = 0.0;
    }
  }

  void calcFmWeights() {
    // calc_w
    if (fabs(z.w) <= train_opt.ftrl.l1_reg_w) {
      fm_param.w = 0.0;
    } else // TODO check
    {
      fm_param.w = -(z.w - utils::sign_a_multiply_b(z.w, train_opt.ftrl.l1_reg_w)) /
          (train_opt.ftrl.l2_reg_w + (train_opt.ftrl.w_beta + std::sqrt(n.w)) / train_opt.ftrl.w_alpha);
    }
    // calc V
    for (int f = 0; f < DIM; ++f) {
      if (n.V[f] > 0) {
        if (fabs(z.V[f]) <= train_opt.ftrl.l1_reg_V) {
          fm_param.V[f] = 0.0;
        } 
        else
        {
          fm_param.V[f] = -(z.V[f] - utils::sign_a_multiply_b(z.V[f], train_opt.ftrl.l1_reg_V)) /
               (train_opt.ftrl.l2_reg_V + (train_opt.ftrl.v_beta + std::sqrt(n.V[f])) / train_opt.ftrl.v_alpha);
        } 
      }
    }
  }
};

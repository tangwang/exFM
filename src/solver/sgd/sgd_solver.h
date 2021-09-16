/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/fea_manager.h"
#include "solver/sgd/sgd_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class SgdSolver : public BaseSolver {
 public:
  SgdSolver(const FeaManager &fea_manager) : BaseSolver(fea_manager), lr(train_opt.sgd.step_size) {}
  virtual ~SgdSolver() {}

  const real_t lr;

  virtual void update(real_t grad) {

    for (auto param_context : backward_params) {
      SgdParamUnit *backward_param = (SgdParamUnit *)param_context.param;
      param_context.mutex->lock();
      real_t xi = 1.0;
      real_t wg = grad * xi;
      backward_param->head.w -= lr *  wg;

      for (int f = 0; f < train_opt.factor_num; ++f) {
        real_t &vf = backward_param->head.V[f];
        real_t vgf = wg * (sum[f]  - vf * xi );
        vf -= lr * vgf;
      }

      param_context.mutex->unlock();
    }
  }
};

/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/fea_manager.h"
#include "solver/ftrl/ftrl_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class FtrlSolver : public BaseSolver {
 public:
  FtrlSolver(const FeaManager &fea_manager) : BaseSolver(fea_manager) {}
  virtual ~FtrlSolver() {}

  virtual void update(real_t grad) {

    for (auto param_context : backward_params) {
      FtrlParamUnit *backward_param = (FtrlParamUnit *)param_context.param;
      param_context.mutex->lock();
      real_t xi = param_context.x;
      grad *= xi;
      real_t w_sigama =
          1 / train_opt.ftrl.w_alpha *
          (std::sqrt(backward_param->n.w + grad * grad) - std::sqrt(backward_param->n.w));

      backward_param->z.w += grad - w_sigama * backward_param->fm_param.w;
      backward_param->n.w += grad * grad;

      for (int f = 0; f < DIM; ++f) {
        real_t vgf = grad * (sum[f]  - backward_param->fm_param.V[f] * xi);
        real_t v_sigma_f =
            1 / train_opt.ftrl.v_alpha * (std::sqrt(backward_param->n.V[f] + vgf * vgf) - std::sqrt(backward_param->n.V[f]));

        backward_param->z.V[f] += vgf - v_sigma_f * backward_param->fm_param.V[f];
        backward_param->n.V[f] += vgf * vgf;
      }

      backward_param->calcFmWeights();

      param_context.mutex->unlock();
    }
  }
};

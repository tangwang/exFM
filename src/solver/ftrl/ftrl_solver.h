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

      real_t & wz = backward_param->multabel_wz();
      real_t & wn = backward_param->multabel_wn();

      real_t xi = 1.0;
      real_t wg = grad * xi;
      real_t ws =
          1 / train_opt.ftrl.w_alpha *
          (sqrt(wn + wg * wg) - sqrt(wn));

      wz += wg - ws * backward_param->head.w;
      wn += wg * wg;

      for (int f = 0; f < train_opt.factor_num; ++f) {
        const real_t &vf = backward_param->head.V[f];
        real_t &vnf = backward_param->multabel_vn(f);
        real_t &vzf = backward_param->multabel_vz(f);
        real_t vgf = wg * (sum[f]  - vf * xi );
        real_t vsf =
            1 / train_opt.ftrl.v_alpha * (sqrt(vnf + vgf * vgf) - sqrt(vnf));

        vzf += vgf - vsf * vf;
        vnf += vgf * vgf;
      }

      backward_param->calc_param();

      param_context.mutex->unlock();
    }
  }
};

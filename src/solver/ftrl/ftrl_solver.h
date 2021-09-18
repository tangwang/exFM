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

      // grad *= xi; //暂时都是离散特征，不支持连续值特征，所以此处关闭
      real_t w_sigama =
          1 / train_opt.ftrl.w_alpha *
          (sqrt(wn + grad * grad) - sqrt(wn));

      wz += grad - w_sigama * backward_param->head.w;
      wn += grad * grad;

      for (int f = 0; f < train_opt.factor_num; ++f) {
        const real_t &vf = backward_param->head.V[f];
        real_t &vnf = backward_param->multabel_vn(f);
        real_t &vzf = backward_param->multabel_vz(f);
        real_t vgf = grad * (sum[f]  - vf * xi );
        real_t v_sigma_f =
            1 / train_opt.ftrl.v_alpha * (sqrt(vnf + vgf * vgf) - sqrt(vnf));

        vzf += vgf - v_sigma_f * vf;
        vnf += vgf * vgf;
      }

      backward_param->calc_param();

      param_context.mutex->unlock();
    }
  }
};

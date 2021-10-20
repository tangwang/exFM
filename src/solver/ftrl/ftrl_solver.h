/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/feat_manager.h"
#include "solver/ftrl/ftrl_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class FtrlSolver : public BaseSolver {
 public:
  FtrlSolver(const FeatManager &feat_manager) : BaseSolver(feat_manager) {}
  virtual ~FtrlSolver() {}

  virtual void update() {

    // TODO FTRL并不需要batchsize，这种通用的处理方法带来很多额外的性能开销。 测试一下，batch_size是否对FTRL的精度有效，没什么作用的话为FTRL专门设计一下Solver
    for (auto & kv : batch_params) {
      ParamNode & param_node = kv.second;
      FMParamUnit & grad = param_node.fm_grad;
      batchReduce(grad, param_node.count);

      FtrlParamUnit *backward_param = (FtrlParamUnit *)param_node.param;
      param_node.mutex->writeLock();
      real_t w_sigama =
          1 / train_opt.ftrl.w_alpha *
          (std::sqrt(backward_param->n.w + grad.w * grad.w) - std::sqrt(backward_param->n.w));

      backward_param->z.w += grad.w - w_sigama * backward_param->fm_param.w;
      backward_param->n.w += grad.w * grad.w;

      for (int f = 0; f < DIM; ++f) {
        real_t vgf = grad.V[f];
        real_t v_sigma_f =
            1 / train_opt.ftrl.v_alpha * (std::sqrt(backward_param->n.V[f] + vgf * vgf) - std::sqrt(backward_param->n.V[f]));

        backward_param->z.V[f] += vgf - v_sigma_f * backward_param->fm_param.V[f];
        backward_param->n.V[f] += vgf * vgf;
      }

      backward_param->calcFmWeights();

      param_node.mutex->unlock();
    }
  }


};

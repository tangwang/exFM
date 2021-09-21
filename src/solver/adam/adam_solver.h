/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/fea_manager.h"
#include "solver/adam/adam_param.h"
#include "solver/base_solver.h"
#include "train/train_opt.h"

class AdamSolver : public BaseSolver {
 public:
  AdamSolver(const FeaManager &fea_manager)
      : BaseSolver(fea_manager),
        lr(train_opt.adam.lr),
        beta1(train_opt.adam.beta1),
        beta2(train_opt.adam.beta2),
        beta1_pow(1.0),
        beta2_pow(1.0),
        weight_decay_w(train_opt.adam.weight_decay_w),
        weight_decay_V(train_opt.adam.weight_decay_V),
        eps(train_opt.adam.eps),
        bias_correct(train_opt.adam.bias_correct != 0) {}

  virtual ~AdamSolver() {}

  virtual void update() {
    // TODO 这里的pow(beta1_pow, t), t是取总步数，该是取该参数更新的次数？

    for (auto &kv : batch_params) {
      ParamContext &param_context = kv.second;
      FMParamUnit grad = param_context.fm_grad;

      switch (train_opt.batch_grad_reduce_type) {
        case TrainOption::BatchGradReduceType_AvgByBatchSize:
          grad /= train_opt.batch_size;
          break;
        case TrainOption::BatchGradReduceType_AvgByOccurrences:
          grad /= param_context.count;
          break;
        case TrainOption::BatchGradReduceType_AvgByOccurrencesSqrt:
          grad /= std::sqrt(param_context.count);
          break;
        default:
          break;
      }

      AdamParamUnit *backward_param = (AdamParamUnit *)param_context.param;
      param_context.mutex->lock();
      // calc fixed_lr
      backward_param->beta1power_t *= beta1;
      backward_param->beta2power_t *= beta2;
      real_t bias_correction1 = (1 - backward_param->beta1power_t);
      real_t bias_correction2 = (1 - backward_param->beta2power_t);
      real_t fixed_lr = lr * std::sqrt(bias_correction2) / bias_correction1;

      // update w
      real_t &w = backward_param->fm_param.w;
      real_t &wm = backward_param->momentum.w;
      real_t &wv = backward_param->variance_m.w;

      wm = beta1 * wm + (1 - beta1) * grad.w;
      wv = beta2 * wv + (1 - beta2) * grad.w * grad.w;

      DEBUG_OUT << "adam_solver: grad:" << grad << " fixed_lr: "
                << " count " << param_context.count << fixed_lr << " wm:" << wm
                << " wv:" << wv << " update:"
                << fixed_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w)
                << endl
                << "fm_param: " << backward_param->fm_param.w << ","
                << backward_param->fm_param.V[0] << ","
                << backward_param->fm_param.V[1] << endl
                << "momentum: " << backward_param->momentum.w << ","
                << backward_param->momentum.V[0] << ","
                << backward_param->momentum.V[1] << endl
                << "variance_m: " << backward_param->variance_m.w << ","
                << backward_param->variance_m.V[0] << ","
                << backward_param->variance_m.V[1] << endl
                << "fm_param.V_0_1 " << backward_param->fm_param.V[0] << ","
                << backward_param->fm_param.V[1] << endl;

      w -= fixed_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w);

      // update V
      for (int f = 0; f < DIM; ++f) {
        real_t &vf = backward_param->fm_param.V[f];
        real_t &vmf = backward_param->momentum.V[f];
        real_t &vvf = backward_param->variance_m.V[f];

        real_t vgf = grad.V[f];

        vmf = beta1 * vmf + (1 - beta1) * vgf;
        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;
        vf -= fixed_lr * (vmf / (std::sqrt(vvf) + eps) + weight_decay_V * vf);
      }
      param_context.mutex->unlock();
    }
  }

  const real_t lr;
  const real_t beta1;
  const real_t beta2;
  real_t beta1_pow;
  real_t beta2_pow;
  const real_t weight_decay_w;
  const real_t weight_decay_V;
  const real_t eps;
  const bool bias_correct;
};

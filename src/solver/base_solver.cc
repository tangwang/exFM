/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "solver/base_solver.h"
#include "feature/fea_manager.h"

real_t Sample::forward() {
  logit = 0.0;
  
  for (int f = 0; f < DIM; ++f) {
    sum[f] = sum_sqr[f] = 0.0;
  }

  size_t forward_params_size = forward_params.size();
  for (size_t i = 0; i < forward_params_size; i++) {
    const ParamContext & ctx = forward_params[i];
    real_t x = ctx.x;
    logit += ctx.param->w * x;
    for (int f = 0; f < DIM; ++f) {
      real_t d = ctx.param->V[f] * x;
      sum[f] += d;
      sum_sqr[f] += d * d;
    }
  }
  real_t sum_factors_score = 0.0;
  for (int f = 0; f < DIM; ++f) {
    sum_factors_score += (sum[f] * sum[f] - sum_sqr[f]);
  }
  sum_factors_score *= 0.5;

  logit += sum_factors_score;

  return logit;
}

void Sample::backward() {
  // 计算整体的梯度:
  // crossEntropyLoss = -log( sigmoid(y * fm_score(x) ) ) ， y = {-1, 1}
  // partitial(loss) / partitial(fm_score(x)) = -y * sigmoid( - fm_score(x) * y )
  real_t exp_y_logit = std::exp(logit * y);
  grad = -y / (1 + exp_y_logit); // TODO 浅层模型是否需要clip gradients
  loss = - std::log(1 - 1/(1+std::max(exp_y_logit, 1e-10)));
  
  // 计算每个fmParamUnit的梯度： partitial(fm_score(x)) / partitial(\theta),  theata = {w_i, V_i1, Vi2, ... Vif} for i in {0, 1, ... N }
  for (auto &param_context : backward_params) {
    real_t xi = param_context.x;
    real_t grad_i = grad * xi;
    FMParamUnit *backward_param = param_context.param;
    param_context.fm_grad.w = grad_i;
    for (int f = 0; f < DIM; ++f) {
      real_t &vf = backward_param->V[f];
      real_t vgf = grad_i * (sum[f] - vf * xi);
      param_context.fm_grad.V[f] = vgf;
    }
  }
}

BaseSolver::BaseSolver(const FeaManager &fea_manager)
    : fea_manager_(fea_manager), batch_size(train_opt.batch_size), sample_idx(0), batch_samples(train_opt.batch_size) {
  for (auto &iter : fea_manager_.dense_feas) {
    dense_feas.push_back(std::move(DenseFeaContext(iter)));
  }
  for (auto &iter : fea_manager_.sparse_feas) {
    sparse_feas.push_back(std::move(SparseFeaContext(iter)));
  }
  for (auto &iter : fea_manager_.varlen_feas) {
    varlen_feas.push_back(std::move(VarlenSparseFeaContext(iter)));
  }
}

int BaseSolver::feedSample(const char *line) {
  // label统一为1， -1的形式
  // y = atoi(line) > 0 ? 1 : -1;
  if (UNLIKELY(*line == 0)) {
    return -1;
  }

  Sample & sample = batch_samples[sample_idx];
  sample.y = line[0] == '1' ? 1 : -1;

  do {
    ++line;
  } while (*line != train_opt.fea_seperator);

  sample.forward_params.clear();
  sample.backward_params.clear();
  for (auto &iter : dense_feas) {
    iter.feedSample(line, sample.forward_params, sample.backward_params);
  }
  for (auto &iter : sparse_feas) {
    iter.feedSample(line, sample.forward_params, sample.backward_params);
  }
  for (auto &iter : varlen_feas) {
    iter.feedSample(line, sample.forward_params, sample.backward_params);
  }
  return 0;
}


#if 0 // 单个样本的sgdm, adam, ftrl参数更新

  void update_by_sgdm() {
    for (auto & param_context : backward_params) {
      real_t grad = param_context.grad;
      real_t xi = param_context.xi;

      SgdmParamUnit *backward_param = (SgdmParamUnit *)param_context.param;
      param_context.mutex->lock();

      real_t & w = backward_param->fm_param.w;
      real_t & wm = backward_param->momentum.w;

      wm = beta1 * wm + (1-beta1) * grad;
      w -= lr * (wm  + w * l2_reg_w);

      for (int f = 0; f < DIM; ++f) {
        real_t &vf = backward_param->fm_param.V[f];
        real_t & vmf = backward_param->momentum.V[f];

        real_t vgf = grad * (sum[f]  - vf * xi );

        vmf = beta1 * vmf + (1-beta1) * vgf;

        vf -= lr * (vmf + vf * l2_reg_V);
      }
      param_context.mutex->unlock();
    }
  }

  virtual void update_by_adam() {
    // TODO 这里的pow(beta1_pow, t), t是取总步数，该是取该参数更新的次数？

    for (auto & param_context : backward_params) {
      real_t grad = param_context.grad;
      real_t xi = param_context.xi;

      AdamParamUnit *backward_param = (AdamParamUnit *)param_context.param;
      param_context.mutex->lock();
      // calc fixed_lr
      backward_param->beta1power_t *= beta1;
      backward_param->beta2power_t *= beta2;
      real_t bias_correction1 = (1 - backward_param->beta1power_t);
      real_t bias_correction2 = (1 - backward_param->beta2power_t);
      real_t fixed_lr = lr * std::sqrt(bias_correction2) / bias_correction1;

      // update w
      real_t & w = backward_param->fm_param.w;
      real_t & wm = backward_param->momentum.w;
      real_t & wv = backward_param->avg_squared.w;

      wm = beta1 * wm + (1-beta1)*grad;
      wv = beta2 * wv + (1-beta2)*grad*grad;

      DEBUG_OUT << "adam_solver: grad:" << grad << " w:" << w << " fixed_lr: " << fixed_lr
                << " wm:" << wm << " wv:" << wv << " update:"
                << fixed_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w) << endl
                << "fm_param: " << backward_param->fm_param.w << "," << backward_param->fm_param.V[0] << "," << backward_param->fm_param.V[1] << endl
                << "momentum: " << backward_param->momentum.w << "," << backward_param->momentum.V[0] << "," << backward_param->momentum.V[1] << endl
                << "avg_squared: " << backward_param->avg_squared.w << "," << backward_param->avg_squared.V[0] << "," << backward_param->avg_squared.V[1] << endl
                << "sum_0_1 " << sum[0] <<"," << sum[1] << endl
                << "fm_param.V_0_1 " << backward_param->fm_param.V[0] <<"," << backward_param->fm_param.V[1] << endl
                << "vgf_0 " << grad * (sum[0]  - backward_param->fm_param.V[0] * xi ) << endl
                << "vgf_1 " << grad * (sum[1]  - backward_param->fm_param.V[1] * xi ) << endl;

      w -= fixed_lr * (wm / (std::sqrt(wv) + eps) + weight_decay_w * w);

      // update V
      for (int f = 0; f < DIM; ++f) {

        real_t &vf = backward_param->fm_param.V[f];
        real_t &vmf = backward_param->momentum.V[f];
        real_t &vvf = backward_param->avg_squared.V[f];

        real_t vgf = grad * (sum[f]  - vf * xi );

        vmf = beta1 * vmf + (1 - beta1) * vgf;
        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;
        vf -= fixed_lr * (vmf / (std::sqrt(vvf) + eps) + weight_decay_V * vf);
      }
      param_context.mutex->unlock();
    }
  }

  void update_by_adam_raw(real_t grad) {
    // TODO 这里的pow(beta1_pow, t), t是取总步数，该是取该参数更新的次数？

    for (auto & param_context : backward_params) {
      real_t grad = param_context.grad;
      real_t xi = param_context.xi;

      AdamParamUnit *backward_param = (AdamParamUnit *)param_context.param;
      param_context.mutex->lock();

      real_t & w = backward_param->fm_param.w;
      real_t & wm = backward_param->momentum.w;
      real_t & wv = backward_param->avg_squared.w;

      wm = beta1 * wm + (1-beta1)*grad;
      wv = beta2 * wv + (1-beta2)*grad*grad;

      real_t corrected_wm = wm;
      real_t corrected_wv = wv;
      if (bias_correct) {
        backward_param->beta1power_t *= beta1;
        backward_param->beta2power_t *= beta2;
        wm /= (1-backward_param->beta1power_t);
        wv /= (1-backward_param->beta2power_t);
      }
      
      w -= lr * (corrected_wm / (std::sqrt(corrected_wv) + eps) + weight_decay_w * w);

      for (int f = 0; f < DIM; ++f) {

        real_t &vf = backward_param->fm_param.V[f];
        real_t &vmf = backward_param->momentum.V[f];
        real_t &vvf = backward_param->avg_squared.V[f];

        real_t vgf = grad * (sum[f]  - vf * xi );

        vmf = beta1 * vmf + (1 - beta1) * vgf;
        vvf = beta2 * vvf + (1 - beta2) * vgf * vgf;

        real_t corrected_vmf = bias_correct ? vmf : (vmf / (1 - beta1_pow));
        real_t corrected_vvf = bias_correct ? vvf : (vvf / (1 - beta2_pow));

        vf -= lr * (corrected_vmf /
               (std::sqrt(corrected_vvf) + eps)  + weight_decay_V * vf);
      }

      param_context.mutex->unlock();
    }
  }


  void update_by_ftrl() {

    // TODO FTRL并不需要batchsize，这种通用的处理方法带来很多额外的性能开销。 测试一下，batch_size是否对FTRL的精度有效，没什么作用的话为FTRL专门设计一下Solver
    for (auto & param_context : backward_params) {
      real_t grad = param_context.grad;
      real_t xi = param_context.xi;

      FtrlParamUnit *backward_param = (FtrlParamUnit *)param_context.param;
      param_context.mutex->lock();
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

#endif

// void BaseSolver::update__by_container(real_t grad) {

//   for (auto param_context : backward_params) {
//     param_context.mutex->lock();
//     param_context.container->update_param(param_context.param, grad);
//     param_context.mutex->unlock();
//   }
// }
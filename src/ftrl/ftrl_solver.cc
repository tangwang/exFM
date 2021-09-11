/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "ftrl/ftrl_solver.h"

FTRLSolver::FTRLSolver(const FeaManager &fea_manager)
    : ISolver(fea_manager),
      sum(train_opt.factor_num),
      bias_container(1),
      USE_BIAS(false) {
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

FTRLSolver::~FTRLSolver() {}

int FTRLSolver::feedSample(const char *line) {
  // label统一为1， -1的形式
  // y = atoi(line) > 0 ? 1 : -1;
  if (UNLIKELY(*line == 0)) {
    return -1;
  }
  y = line[0] == '1' ? 1 : -1;

  do {
    ++line;
  } while (*line != train_opt.fea_seperator);

  forward_params.clear();
  backward_params.clear();
  for (auto &iter : dense_feas) {
    iter.feedSample(line, forward_params, backward_params);
  }
  for (auto &iter : sparse_feas) {
    iter.feedSample(line, forward_params, backward_params);
  }
  for (auto &iter : varlen_feas) {
    iter.feedSample(line, forward_params, backward_params);
  }
  return 0;
}

void FTRLSolver::train_fm_flattern(int &out_y, real_t &out_logit,
                                   bool only_predict) {
  real_t logit = 0.0;
  real_t sum_sqr = 0.0;
  real_t d = 0.0;
  real_t mult = 0.0;

  if (USE_BIAS) {
    FtrlParamUnit *bias = bias_container.get(0);
    forward_params.push_back(bias);
    backward_params.push_back(bias);
  }
  for (auto param_context : backward_params) {
    FtrlParamUnit *backward_param = param_context.param;
    param_context.mutex->lock();
    backward_param->calc_param();
    logit += backward_param->w;
    param_context.mutex->unlock();
  }

  for (int f = 0; f < train_opt.factor_num; ++f) {
    sum[f] = sum_sqr = 0.0;
    for (auto param_context : backward_params) {
      FtrlParamUnit *backward_param = param_context.param;
      param_context.mutex->lock();
      d = param_context.param->v(f);
      sum[f] += d;
      sum_sqr += d * d;
      param_context.mutex->unlock();
    }

    logit += 0.5 * (sum[f] * sum[f] - sum_sqr);
    // if (fabs(logit) > 100)
    //     printf("AAA   logit, d, sum[f], sum_sqr %f  %f %f  %f   \n ",
    //     logit, d, sum[f], sum_sqr );
  }
  logit = logit;
  out_y = y;
  out_logit = logit;
  if (only_predict) {
    return;
  }

  mult = y * (1 / (1 + exp(-logit * y)) - 1);
  // printf("y, logit, mult   %d %f %f \n ", y, logit, mult);

  // if (fabs(logit) > 100)
  //   printf(" BBBBBBBBBBBB %f  %f  %f  %f  %f  %f \n", mult, logit, sum[1],
  //          sum[2], sum[3], sum_sqr);
  for (auto param_context : backward_params) {
    FtrlParamUnit *backward_param = param_context.param;
    param_context.mutex->lock();
    real_t xi = 1.0;
    real_t wg = mult * xi;
    real_t ws = 1 / train_opt.ftrl.w_alpha *
                (sqrt(backward_param->wn + wg * wg) - sqrt(backward_param->wn));

    backward_param->wz += wg - ws * backward_param->w;
    backward_param->wn += wg * wg;

    for (int f = 0; f < train_opt.factor_num; ++f) {
      const real_t &vf = backward_param->v(f);
      real_t &vnf = backward_param->multabel_vn(f);
      real_t &vzf = backward_param->multabel_vz(f);
      real_t vgf = mult * (sum[f] * xi - vf * xi * xi);
      real_t vsf =
          1 / train_opt.ftrl.v_alpha * (sqrt(vnf + vgf * vgf) - sqrt(vnf));

      vzf += vgf - vsf * vf;
      vnf += vgf * vgf;
    }
    // if (fabs(logit) > 500)
    //   printf("CCC dense_feas  wz wn vz vf %f %f   %f %f    \n ",
    //          backward_param->wz, backward_param->wn,
    //          backward_param->multabel_vz(3), backward_param->multabel_vn(3));
    param_context.mutex->unlock();
  }
}

void FTRLSolver::train(int &out_y, real_t &out_logit, bool only_predict) {
  real_t logit = 0.0;
  real_t sum_sqr = 0.0;
  real_t d = 0.0;
  real_t mult = 0.0;

  if (USE_BIAS) {
    FtrlParamUnit *bias = bias_container.get(0);
    forward_params.push_back(bias);
    backward_params.push_back(bias);
  }
  for (auto param_context : forward_params) {
    logit += param_context.param->w;
  }

  for (int f = 0; f < train_opt.factor_num; ++f) {
    sum[f] = sum_sqr = 0.0;
    for (auto param_context : forward_params) {
      d = param_context.param->v(f);
      sum[f] += d;
      sum_sqr += d * d;
    }

    logit += 0.5 * (sum[f] * sum[f] - sum_sqr);
    // if (fabs(logit) > 100)
    //     printf("AAA   logit, d, sum[f], sum_sqr %f  %f %f  %f   \n ",
    //     logit, d, sum[f], sum_sqr );
  }
  logit = logit;
  out_y = y;
  out_logit = logit;
  if (only_predict) {
    return;
  }

  mult = y * (1 / (1 + exp(-logit * y)) - 1);
  // printf("y, logit, mult   %d %f %f \n ", y, logit, mult);

  // if (fabs(logit) > 100)
  //   printf(" BBBBBBBBBBBB %f  %f  %f  %f  %f  %f \n", mult, logit, sum[1],
  //          sum[2], sum[3], sum_sqr);
  for (auto param_context : backward_params) {
    FtrlParamUnit *backward_param = param_context.param;
    param_context.mutex->lock();
    real_t xi = 1.0;
    real_t wg = mult * xi;
    real_t ws = 1 / train_opt.ftrl.w_alpha *
                (sqrt(backward_param->wn + wg * wg) - sqrt(backward_param->wn));

    backward_param->wz += wg - ws * backward_param->w;
    backward_param->wn += wg * wg;

    for (int f = 0; f < train_opt.factor_num; ++f) {
      const real_t &vf = backward_param->v(f);
      real_t &vnf = backward_param->multabel_vn(f);
      real_t &vzf = backward_param->multabel_vz(f);
      real_t vgf = mult * (sum[f] * xi - vf * xi * xi);
      real_t vsf =
          1 / train_opt.ftrl.v_alpha * (sqrt(vnf + vgf * vgf) - sqrt(vnf));

      vzf += vgf - vsf * vf;
      vnf += vgf * vgf;
    }
    // if (fabs(logit) > 500)
    //   printf("CCC dense_feas  wz wn vz vf %f %f   %f %f    \n ",
    //          backward_param->wz, backward_param->wn,
    //          backward_param->multabel_vz(3), backward_param->multabel_vn(3));
    param_context.mutex->unlock();
  }
}

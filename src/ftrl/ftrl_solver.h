/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "feature/fea_manager.h"
#include "ftrl/ftrl_param.h"
#include "train/train_opt.h"
#include "train/solver_interface.h"

class FTRLSolver : public ISolver {
 public:
  FTRLSolver(const FeaManager &fea_manager);
  ~FTRLSolver();

  const bool
      USE_BIAS;  // 用不用bias，AUC很接近，而且bias更新频繁，影响性能，默认不开启。

  vector<DenseFeaContext> dense_feas;
  vector<SparseFeaContext> sparse_feas;
  vector<VarlenSparseFeaContext> varlen_feas;
  FtrlParamContainer bias_container;

  // 填充样本后收集的param_list. TODO 后期要支持连续特征，包括 常量embedding特征
  // 必须每个维度作为连续特征用进来，需要存储每个位置的x，考虑改成vector<pair<real_t,
  // FtrlParamUnit *>>
  vector<ParamContext> forward_params;
  vector<ParamContext> backward_params;


  int y;
  real_t logit;
  vector<real_t> sum;

  int feedSample(const char *line);

  void train(int &out_y, real_t &out_logit, bool only_predict = false);
  void train_fm_flattern(int &out_y, real_t &out_logit,
                         bool only_predict = false);

};

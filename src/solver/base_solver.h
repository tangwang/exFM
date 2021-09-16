/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "utils/base.h"
#include "solver/parammeter_interface.h"
#include "feature/fea_manager.h"


class BaseSolver {
 public:
  BaseSolver(const FeaManager &fea_manager);

  virtual ~BaseSolver() {}

  int feedSample(const char *line);

  void train(int &out_y, real_t &out_logit);

  real_t predict();

  virtual void update(real_t grad) = 0;

  // // 不用这个，用多态的update
  // void update__by_container(real_t grad);

  vector<DenseFeaContext> dense_feas;
  vector<SparseFeaContext> sparse_feas;
  vector<VarlenSparseFeaContext> varlen_feas;

  // 填充样本后收集的param_list. TODO 后期要支持连续特征，包括 常量embedding特征
  // 必须每个维度作为连续特征用进来，需要存储每个位置的x，考虑改成vector<pair<real_t,
  // FtrlParamUnit *>>
  vector<ParamContext> forward_params;
  vector<ParamContext> backward_params;

  vector<real_t> sum;
  real_t logit;
  int y;

 protected:
  const FeaManager &fea_manager_;
};

/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include "utils/base.h"

class ISolver {
 public:
  ISolver(const FeaManager &fea_manager) : fea_manager_(fea_manager) {}
  virtual ~ISolver() {}

  virtual int feedSample(const char *line) = 0;

  virtual void train(int &out_y, real_t &out_logit, bool only_predict = false) = 0;
  virtual void train_fm_flattern(int &out_y, real_t &out_logit,
                         bool only_predict = false) = 0;

protected:
  const FeaManager &fea_manager_;

};

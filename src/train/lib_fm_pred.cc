/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include "solver/adagrad/adagrad_solver.h"
#include "solver/adam/adam_solver.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/rmsprop/rmsprop_solver.h"
#include "solver/sgdm/sgdm_solver.h"
#include "solver/solver_factory.h"
#include "train/shulffer.h"
#include "train/train_worker.h"

extern "C" {

FeatManager feat_manager;
BaseSolver* solver = NULL;

int fm_model_init() {
  if (!train_opt.parse_cfg_and_cmdlines(0, NULL)) {
    cerr << "parse args faild, exit" << endl;
    return -1;
  }

  assert(!train_opt.feature_config_path.empty());
  assert(access(train_opt.feature_config_path.c_str(), F_OK) != -1);
  if (!feat_manager.loadByFeatureConfig(train_opt.feature_config_path)) {
    cerr << "init feature manager faild, check config file "
         << train_opt.feature_config_path << ". exit" << endl;
    return -1;
  }

  train_opt.solver = "pred";
  solver = new BaseSolver(feat_manager);

  return 0;
}

real_t pred(const string& line) {
  int y;
  real_t score;
  solver->test(line, y, score);
  return score;
}

void fm_pred(char* str, char* ret, int ret_len) {
  vector<string> lines;
  vector<real_t> scores;
  utils::split_string(str, train_opt.feat_seperator, lines);
  for (const auto& line : lines) {
    scores.push_back(pred(line));
  }
  std::stringstream sstream;
  string str_ret;
  sstream << scores;
  sstream >> str_ret;
  strncpy(ret, str_ret.c_str(), std::min((size_t)ret_len, str_ret.size()));
}

}
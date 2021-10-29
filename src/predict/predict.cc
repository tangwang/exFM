/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include "train/train_worker.h"
#include "train/shulffer.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/solver_factory.h"
#include "solver/ftrl/ftrl_solver.h"
#include "solver/adam/adam_solver.h"
#include "solver/sgdm/sgdm_solver.h"
#include "solver/adagrad/adagrad_solver.h"
#include "solver/rmsprop/rmsprop_solver.h"


int main(int argc, char *argv[]) {
  srand(time(NULL));

  if (!train_opt.parse_cfg_and_cmdlines(argc, argv)) {
    cerr << "parse args faild, exit" << endl;
    return -1;
  }

  train_opt.solver = "pred";

  cin.sync_with_stdio(false);

// 如果是csv格式，解析头行
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    std::getline(cin, train_opt.csv_columns);
  }

  // init trainning workers
  FeatManager feat_manager;
  assert(!train_opt.feature_config_path.empty());
  assert(access(train_opt.feature_config_path.c_str(), F_OK) != -1);
  if (!feat_manager.loadByFeatureConfig(train_opt.feature_config_path)) {
    cerr << "init feature manager faild, check config file " << train_opt.feature_config_path << ". exit" << endl;
    return -1;
  }

  BaseSolver solver(feat_manager);

  int y;
  real_t score;
  string line;
  cout << "y" << train_opt.feat_seperator << "score" << train_opt.feat_seperator << "input_line" << endl;
  while (std::getline(std::cin, line)) {
    solver.test(line, y, score);
    int pred = score > 0.0 ? 1 : 0;
    cout << pred << train_opt.feat_seperator << score << endl;
  }

  return 0;
}

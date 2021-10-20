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
  // init train input stream
  std::istream *input_stream = NULL;
  std::istream *input_file_stream = NULL;
  if (!train_opt.train_path.empty()) {
    input_file_stream = new ifstream(train_opt.train_path);
    input_stream = input_file_stream;
    if (!(*input_file_stream)) {
      cerr << "train file open filed " << endl;
      delete input_file_stream;
      return -1;
    }
  } else {
    cin.sync_with_stdio(false);
    input_stream = &std::cin;
  }

// 如果是csv格式，解析头行
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    string csv_header;
    std::getline(*input_stream, csv_header);
    utils::split_string(csv_header, train_opt.feat_seperator, train_opt.csv_columns);
  }

  // init trainning workers
  FeatManager feat_manager;
  assert(!train_opt.feature_config_path.empty());
  assert(access(train_opt.feature_config_path.c_str(), F_OK) != -1);
  if (!feat_manager.loadByFeatureConfig(train_opt.feature_config_path)) {
    cerr << "init feature manager faild, check config file " << train_opt.feature_config_path << ". exit" << endl;
    return -1;
  }

  train_opt.solver = "pred";
  BaseSolver solver(feat_manager);

  int y;
  real_t score;
  string line;
  while (std::getline(*input_stream, line)) {
    solver.test(line, y, score);
    cout << y << " " << score << " " << line << endl;
  }

  return 0;
}

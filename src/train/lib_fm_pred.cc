/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#include "feature/feat_manager.h"
#include "train/train_worker.h"
/*
https://www.cnblogs.com/alex96/p/11363424.html

*/
extern "C" {

FeatManager feat_manager;
BaseSolver* solver = NULL;

/*
 need config/train.conf
@return: 0 : success;  other : faild
*/
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
    return -2;
  }

  train_opt.solver = "pred";
  solver = new BaseSolver(feat_manager);

  return 0;
}

void fm_model_release() {
  if (solver) delete solver;
}

real_t pred(const string& line) {
  int y;
  real_t score;
  solver->test(line, y, score);
  return score;
}

/*
@param input_str : support csv / libsvm formart
@param output_str : output memory allocated by caller
@param output_len : memory size of output_str
@return: 0 : success;  other : faild
*/
int fm_pred(char* input_str, char* output_str, int output_len) {
  vector<string> lines;
  vector<real_t> scores;
  utils::split_string(input_str, train_opt.feat_seperator, lines);
  for (const auto& line : lines) {
    scores.push_back(pred(line));
  }
  size_t out_num = scores.size();
  if (out_num == 0) {
    output_str[0] = '\0';
    return 0;
  }
  size_t write_offset = 0;
  write_offset += snprintf(output_str, output_len, "%4f", scores[0]);
  for (size_t i = 1; i < out_num; i++) {
    size_t rest_len = (size_t)output_len - write_offset;
    if (rest_len < 5) {
      return -1;
    }
    write_offset += snprintf(output_str + write_offset, rest_len,",%4f", scores[i]);
  }
  return 0;
}

}
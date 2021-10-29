/**
 *  Copyright (c) 2021 by exFM Contributors
 * 该部分代码未测试，请不要使用
 */
#include "predict/lib_fm_pred.h"
#include "feature/feat_manager.h"
#include "train/train_worker.h"

FmModel::FmModel() {}

int FmModel::init(const char * config_path, const char* input_columns) {
  train_opt.config_file_path = config_path;
  train_opt.solver = "pred";

  // 如果是csv格式，解析头行
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    if ((input_columns == NULL)) {
      cerr << "need specify input_columns for your csv data" << endl;
      return -1;
    }
    train_opt.csv_columns = input_columns;
  }

  if (!train_opt.parse_cfg_and_cmdlines(0, NULL)) {
    cerr << "parse args faild, exit" << endl;
    return -2;
  }

  assert(!train_opt.feature_config_path.empty());
  assert(access(train_opt.feature_config_path.c_str(), F_OK) != -1);

  feat_manager = std::make_shared<FeatManager>();

  if (!feat_manager->loadByFeatureConfig(train_opt.feature_config_path)) {
    cerr << "init feature manager faild, check config file "
         << train_opt.feature_config_path << ". exit" << endl;
    return -3;
  }

  return 0;
}

FmModel::~FmModel() {}

FmPredictInstance* FmModel::getFmPredictInstance() {
  return new FmPredictInstance(*feat_manager);
}

FmPredictInstance::FmPredictInstance(FeatManager& feat_manager) {
  solver = std::make_shared<BaseSolver>(feat_manager);
}

FmPredictInstance::~FmPredictInstance() {}

int FmPredictInstance::fm_pred(const vector<string>& p_lines, vector<double>& p_scores) {
  for (const auto& line : p_lines) {
    p_scores.push_back(predict_line(line));
  }
  return 0;
}

int FmPredictInstance::fm_pred(char* input_str, char* output_str,
                               int output_len) {
  lines.clear();
  scores.clear();
  utils::split_string(input_str, train_opt.feat_seperator, lines);
  for (const auto& line : lines) {
    scores.push_back(predict_line(line));
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
    write_offset +=
        snprintf(output_str + write_offset, rest_len, ",%4f", scores[i]);
  }
  return 0;
}

double FmPredictInstance::predict_line(const string& line) {
  int y;
  real_t score;
  solver->test(line, y, score);
  return score;
}

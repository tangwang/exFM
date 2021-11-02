/**
 *  Copyright (c) 2021 by exFM Contributors
 * 该部分代码未测试，请不要使用
 */
#include "predict/lib_fm_pred.h"
#include "feature/feat_manager.h"
#include "train/train_worker.h"

FmModel::FmModel() {}

int FmModel::init(const char * config_path) {
  train_opt.config_file_path = config_path;
  train_opt.solver = "pred";

  // 如果是csv格式，解析头行
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    if (train_opt.csv_columns.empty()) {
      cerr << "need set csv_columns in your config" << endl;
      return -1;
    }
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

FmPredictInstance* FmModel::createFmPredictInstance() {
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


// 为java或者c项目提供的调用方式
extern "C" {

// @param config_path 配置文件地址
FmModel* fmModelCreate(const char* config_path) {
  FmModel* fm_model = new FmModel;
  int ret = fm_model->init(config_path);
  if (ret != 0) {
    delete fm_model;
    return NULL;
  }
  return fm_model;
}

void fmModelRelease(FmModel* fm_model) {
  if (fm_model) delete fm_model;
}

FmPredictInstance * fmPredictInstanceCreate(FmModel* fm_model) {
  return fm_model->createFmPredictInstance();
}

void fmPredictInstanceRelease(FmPredictInstance* fm_instance) {
  if (fm_instance) delete fm_instance;
}

/*
@param input_str : support csv / libsvm formart
@param output_str : output memory allocated by caller
@param output_len : memory size of output_str
@return: 0 : success;  other : faild
*/
int fmPredict(FmPredictInstance * fm_instance, char* input_str, char* output_str, int output_len) {
  assert(fm_instance != NULL);
  return fm_instance->fm_pred(input_str, output_str, output_len);
}

}
/**
 *  Copyright (c) 2021 by exFM Contributors
 * 该部分代码未测试，请不要使用
 */
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <vector>
#include <iostream>
#include <memory>

/*
 * usage:
// config/train.conf必须配置的参数有：
// data_formart     数据格式, csv/libsvm
// feat_sep         域分隔符
// feat_values_sep  序列特征分隔符
// feat_cfg         特征处理配置
// mf               模型格式
// im               模型地址
// 如果是csv格式，需在第二个参数指定input_columns
FmModel fm_model;
int model_init_ret = fm_model.init("config/train.conf", "item_id,chanel,item_tags,item_clicks,item_price,user_click_list,user_age");
if (0 != model_init_ret) {
  std::cout << " model init error : " << ret << std::endl;
}

// 每个线程可以创建自己的instance
FmPredictInstance* fm_instance = fm_model.getFmPredictInstance();

// 调用预估
const char* intput_str =
    "123,aaa,信托|记账|酒店,2521,0.3,342|5212|839,24\n"
    "3423,bcd,培训|租房,342,1.2,44|3422|34|8,33\n";
char predict_output[10240];
fm_instance->fm_pred(intput_str, predict_output, sizeof(predict_output));

或者
vector<string> input_vec;
input_vec.push_back("3423,bcd,培训|租房,342,1.2,44|3422|34|8,33");
input_vec.push_back("123,aaa,信托|记账|酒店,2521,0.3,342|5212|839,24");
input_vec.push_back("3423,bcd,培训|租房,342,1.2,44|3422|34|8,33");

vector<double> scores
fm_instance->fm_pred(input_vec, scores);
 */

class FmPredictInstance;
class BaseSolver;
class FeatManager;

class FmModel {
 public:
  // @param config_path 配置文件地址
  // @param input_columns:  如果数据为csv格式，在这里指定csv的表头内容
  FmModel();

  int init(const char * config_path, const char* input_columns = NULL);

  ~FmModel();

  FmPredictInstance* getFmPredictInstance();

 private:
  std::shared_ptr<FeatManager> feat_manager;
};


class FmPredictInstance {
 public:
  FmPredictInstance(FeatManager& feat_manager);
  ~FmPredictInstance();

  int fm_pred(const std::vector<std::string>& p_lines, std::vector<double>& p_scores);

  /*
  @param input_str : support csv / libsvm formart
  @param output_str : output memory allocated by caller
  @param output_len : memory size of output_str
  @return: 0 : success;  other : faild
  */
  int fm_pred(char* input_str, char* output_str, int output_len);

 private:
  std::shared_ptr<BaseSolver> solver;
  std::vector<std::string> lines;
  std::vector<double> scores;

  double predict_line(const std::string& line);

};


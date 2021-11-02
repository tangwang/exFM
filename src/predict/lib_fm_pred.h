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
// 如果是csv格式，必须通过csv_columns参数设定列名
FmModel fm_model;
int model_init_ret = fm_model.init("config/train.conf");
if (0 != model_init_ret) {
  std::cout << " model init error : " << ret << std::endl;
}

// 每个线程可以创建自己的instance
FmPredictInstance* fm_instance = fm_model.createFmPredictInstance();

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
  FmModel();

  int init(const char * config_path);

  ~FmModel();

  FmPredictInstance* createFmPredictInstance();

 private:
  std::shared_ptr<FeatManager> feat_manager;
};


// 每个线程/进程创建自己的用于predict的instance
class FmPredictInstance {
 public:
  FmPredictInstance(FeatManager& feat_manager);
  ~FmPredictInstance();

  int fm_pred(const std::vector<std::string>& p_lines, std::vector<double>& p_scores);

  /*
  @param input_str : support csv / libsvm formart
@param output_str : output memory allocated by caller, will fill with scorelist joind by ','
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

// 为java或者c项目提供的调用方式
extern "C" {

//创建模型
FmModel* fmModelCreate(const char* config_path);
void fmModelRelease(FmModel* fm_model);


// 每个线程创建自己的用于predict的instance
FmPredictInstance * fmPredictInstanceCreate(FmModel* fm_model);
void fmPredictInstanceRelease(FmPredictInstance* fm_instance);

//调用预估
/*
@param input_str : support csv / libsvm formart
@param output_str : output memory allocated by caller, will fill with scorelist joind by ','
@param output_len : memory size of output_str
@return: 0 : success;  other : faild
*/
int fmPredict(FmPredictInstance * fm_instance, char* input_str, char* output_str, int output_len);

}
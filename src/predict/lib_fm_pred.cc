// Start of Selection
/**
 *  Copyright (c) 2021 by exFM Contributors
 * 该部分代码未测试，请不要使用
 */
#include "predict/lib_fm_pred.h"
#include "feature/feat_manager.h"
#include "train/train_worker.h"
#include <exception>
#include <new>

FmModel::FmModel() {}

int FmModel::init(const char * config_path) {
  std::cout << "Initializing FmModel with config path: " << config_path << std::endl;
  train_opt.config_file_path = config_path;
  train_opt.solver = "pred";

  // 如果是csv格式，解析头行
  if (train_opt.data_formart == TrainOption::DataFormart_CSV) {
    std::cout << "Data format is CSV. Checking csv_columns configuration." << std::endl;
    if (train_opt.csv_columns.empty()) {
      std::cerr << "Error: csv_columns not set in config." << std::endl;
      return -1;
    }
  }

  std::cout << "Parsing configuration and command line arguments." << std::endl;
  int argc = 1;
  char *argv[] = {const_cast<char*>("lib_fm_pred")};
  if (!train_opt.parse_cfg_and_cmdlines(argc, argv)) {
    std::cerr << "Error: Failed to parse arguments." << std::endl;
    return -2;
  }
  
  char current_path[1024];
  if (getcwd(current_path, sizeof(current_path)) != NULL) {
    std::cout << "Current working directory: " << current_path << std::endl;
  } else {
    std::cerr << "Error: Unable to get current working directory." << std::endl;
    return -4;
  }

  std::cout << "Checking feature configuration path: " << train_opt.feature_config_path << std::endl;
  assert(!train_opt.feature_config_path.empty());
  assert(access(train_opt.feature_config_path.c_str(), F_OK) != -1);

  std::cout << "Creating feature manager." << std::endl;
  feat_manager = std::make_shared<FeatManager>();

  std::cout << "Loading feature manager with configuration: " << train_opt.feature_config_path << std::endl;
  if (!feat_manager->loadByFeatureConfig(train_opt.feature_config_path)) {
    std::cerr << "Error: Failed to initialize feature manager. Check config file: "
              << train_opt.feature_config_path << std::endl;
    return -3;
  }

  std::cout << "FmModel initialized successfully." << std::endl;
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

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

int FmPredictInstance::fm_pred(const vector<string>& p_lines, vector<double>& p_scores) {
  for (const auto& line : p_lines) {
    p_scores.push_back(sigmoid(predict_line(line)));
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
    if (output_len > 0) {
      output_str[0] = '\0';
    }
    return 0;
  }
  size_t write_offset = 0;
  write_offset += snprintf(output_str + write_offset, output_len - write_offset, "%4f", scores[0]);
  for (size_t i = 1; i < out_num; i++) {
    size_t rest_len = (size_t)output_len - write_offset;
    if (rest_len < 5) {
      return -1;
    }
    write_offset += snprintf(output_str + write_offset, rest_len, ",%4f", scores[i]);
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
  printf("Creating FmModel with config path: %s\n", config_path);
  FmModel* fm_model = new (std::nothrow) FmModel; 
  if (!fm_model) {
    printf("Failed to allocate memory for FmModel.\n");
    return NULL;
  }
  int ret = fm_model->init(config_path);
  if (ret != 0) {
    printf("Failed to initialize FmModel with config path: %s, return code: %d\n", config_path, ret);
    delete fm_model;
    return NULL;
  }
  printf("Successfully created FmModel with config path: %s\n", config_path);
  return fm_model;
}

void fmModelRelease(FmModel* fm_model) {
  if (fm_model) {
    delete fm_model;
  }
}

FmPredictInstance * fmPredictInstanceCreate(FmModel* fm_model) {
  if (!fm_model) {
    printf("fmPredictInstanceCreate called with NULL fm_model.\n");
    return NULL;
  }
  try {
    FmPredictInstance* instance = fm_model->createFmPredictInstance();
    if (!instance) {
      printf("Failed to create FmPredictInstance.\n");
      return NULL;
    }
    return instance;
  } catch (const std::bad_alloc& e) {
    printf("Memory allocation failed in fmPredictInstanceCreate: %s\n", e.what());
    return NULL;
  } catch (const std::exception& e) {
    printf("Exception in fmPredictInstanceCreate: %s\n", e.what());
    return NULL;
  } catch (...) {
    printf("Unknown exception in fmPredictInstanceCreate.\n");
    return NULL;
  }
}

void fmPredictInstanceRelease(FmPredictInstance* fm_instance) {
  if (fm_instance) {
    delete fm_instance;
  }
}

/*
@param input_str : support csv / libsvm format
@param output_str : output memory allocated by caller
@param output_len : memory size of output_str
@return: 0 : success;  other : failed
*/
int fmPredict(FmPredictInstance * fm_instance, char* input_str, char* output_str, int output_len) {
  if (!fm_instance) {
    printf("fmPredict called with NULL fm_instance.\n");
    return -1;
  }
  return fm_instance->fm_pred(input_str, output_str, output_len);
}

int fmPredictBatch(FmPredictInstance* fm_instance, const char** input_strs, int input_count, double* output_scores, int print_debug_info) {
  if (!fm_instance) {
    printf("fmPredictBatch called with NULL fm_instance.\n");
    return -1;
  }
  try {
    vector<string> input_vec;
    input_vec.reserve(input_count);
    for (int i = 0; i < input_count; ++i) {
      input_vec.emplace_back(input_strs[i]);
    }
    vector<double> scores;
    int ret = fm_instance->fm_pred(input_vec, scores);
    if (ret == 0) {
      for (int i = 0; i < input_count; ++i) {
        output_scores[i] = scores[i];
      }
    }
    if (print_debug_info != 0) {
      // Print detailed log to standard output
      printf("fmPredictBatch called with %d input strings\n", input_count);
      for (int i = 0; i < input_count; ++i) {
        printf("Input string %d: %s\n", i, input_strs[i]);
      }
      printf("Prediction result: %d\n", ret);
      if (ret == 0) {
        for (int i = 0; i < input_count; ++i) {
          printf("Score %d: %f\n", i, scores[i]);
        }
      }
    }
    return ret;
  } catch (const std::bad_alloc& e) {
    printf("Memory allocation failed in fmPredictBatch: %s\n", e.what());
    return -1;
  } catch (const std::exception& e) {
    printf("Exception in fmPredictBatch: %s\n", e.what());
    return -1;
  } catch (...) {
    printf("Unknown exception in fmPredictBatch.\n");
    return -1;
  }
}

}
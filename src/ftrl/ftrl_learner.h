/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <atomic>
#include <thread>

#include "feature/fea_manager.h"
#include "train/evalution.h"
#include "utils/busy_consumer_queue.h"

using std::thread;
using utils::BusyConsumerQueue;

/**
 * 每个线程用一个自己的FTRLLearner
 */
class FTRLLearner {
 public:
  FTRLLearner(const FeaManager &fea_manager, const char *task_name,
              int task_id);
  ~FTRLLearner();

  ////////////////
  // thread
  void StartTrainLoop() {
    thread_handler = std::thread(&FTRLLearner::task_process_thread_loop, this);
  }
  void StartValidationLoop(std::ifstream &input_stream) {
    thread_handler = std::thread(&FTRLLearner::validation_thread_loop, this,
                                 std::ref(input_stream));
  }
  void Join() { thread_handler.join(); }
  void Stop() { stop_flag = true; }
  bool TryPush(const string &v) { return task_queue.TryPush(v); }
  void WaitAndPush(const string &v) { task_queue.WaitAndPush(v); }

  BusyConsumerQueue<string> task_queue;
  int task_id_;
  std::thread thread_handler;

  std::atomic<bool> stop_flag;
  string task_name_;
  ///////////////////
  const bool
      USE_BIAS;  // 用不用bias，AUC很接近，而且bias更新频繁，影响性能，默认不开启。

  vector<DenseFeaContext> dense_feas;
  vector<SparseFeaContext> sparse_feas;
  vector<VarlenSparseFeaContext> varlen_feas;
  ParamContainer bias_container;

  // 填充样本后收集的param_list. TODO 后期要支持连续特征，包括 常量embedding特征
  // 必须每个维度作为连续特征用进来，需要存储每个位置的x，考虑改成vector<pair<real_t,
  // FTRLParamUnit *>>
  vector<ParamContext> forward_params;
  vector<ParamContext> backward_params;

  const FeaManager &fea_manager_;
  Evalution eval;

  int y;
  real_t logit;
  vector<real_t> sum;

  void DumpEvalInfo() { eval.output(task_name_.c_str()); }

  int feedRawData(const char *line);

  void train(bool only_predict = false);
  void train_fm_flattern(bool only_predict = false);

  void predict();

  void backward();

  void CollectEvalInfo(Evalution &collect_to) { collect_to += eval; }

 private:
  void task_process_thread_loop() {
    std::vector<string> local_task_queue;
    int counter_for_evalution = 0;
    const int n_sample_per_output = train_opt.n_sample_per_output;
    do {
      task_queue.FeachAll(local_task_queue);
      int task_queue_size = local_task_queue.size();

      for (int i = 0; i < task_queue_size; i++) {
        feedRawData(local_task_queue[i].c_str());
        train_fm_flattern();
      }

      local_task_queue.clear();
      counter_for_evalution += task_queue_size;
      if (counter_for_evalution >= n_sample_per_output) {
        counter_for_evalution = 0;
        DumpEvalInfo();
      }
    } while (!stop_flag);
  }

  void validation_thread_loop(std::ifstream &input_stream) {
    string line_buff;
    do {
      sleep(train_opt.time_interval_of_validation);
      while (std::getline(input_stream, line_buff)) {
        feedRawData(line_buff.c_str());
        train_fm_flattern(true);
      }
      DumpEvalInfo();
      input_stream.clear();
      input_stream.seekg(0);
    } while (!stop_flag);
    // 训练完成后再最后跑一遍测试集
    while (std::getline(input_stream, line_buff)) {
      feedRawData(line_buff.c_str());
      train_fm_flattern(true);
    }
    DumpEvalInfo();
  }
};

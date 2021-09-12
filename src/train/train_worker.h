/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <atomic>
#include <thread>
#include "train/evalution.h"
#include "utils/busy_consumer_queue.h"
#include "train/solver_interface.h"

using std::thread;
using utils::BusyConsumerQueue;

class TrainWorker {
 public:
  TrainWorker(const char *task_name, int task_id)
      : stop_flag(false),
        task_id_(task_id),
        solver(NULL),
        task_queue(train_opt.task_queue_size) {
    task_name_ = string(task_name) + string("_") + std::to_string(task_id);
  }
  ~TrainWorker() {
    if (solver) {
      delete solver;
    }
  }

  void RegisteSolver(ISolver * _solver) {
    if (solver) {
      delete solver;
    }
    solver = _solver;
  }

  ////////////////
  // thread
  void StartTrainLoop() {
    thread_handler = std::thread(&TrainWorker::task_process_thread_loop, this);
  }
  void StartValidationLoop(std::ifstream &input_stream) {
    thread_handler = std::thread(&TrainWorker::validation_thread_loop, this,
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

  Evalution eval;

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
    const bool verbose_debug = train_opt.verbose > 1;

    int y;
    real_t logit;
    do {
      task_queue.FeachAll(local_task_queue);
      int task_queue_size = local_task_queue.size();
      if (verbose_debug)
        std::cout << "task " << task_name_ << " fetched " << task_queue_size
                  << " lines" << std::endl;

      for (int i = 0; i < task_queue_size; i++) {
        solver->feedSample(local_task_queue[i].c_str());
        // train_fm_flattern(y, logit);
        solver->train(y, logit);
        eval.add(y, logit);
      }

      local_task_queue.clear();
      counter_for_evalution += task_queue_size;
      if (counter_for_evalution >= n_sample_per_output) {
        counter_for_evalution = 0;
        eval.output(task_name_.c_str());
      }
    } while (!stop_flag);
  }

  void validation_thread_loop(std::ifstream &input_stream) {
    string line_buff;
    int sleep_seconds = 0;
    const bool verbose_debug = train_opt.verbose > 1;
    int y;
    real_t logit;
    do {
      sleep(2);
      sleep_seconds += 2;
      if (stop_flag)
        break;
      else if (sleep_seconds < train_opt.time_interval_of_validation)
        continue;
      else
        sleep_seconds = 0;
      if (verbose_debug)
        std::cout << "validation thread begin predict... " << std::endl;

      while (std::getline(input_stream, line_buff)) {
        solver->feedSample(line_buff.c_str());
        // solver->train_fm_flattern(y, logit, true);
        solver->train(y, logit, true);
        eval.add(y, logit);
      }
      eval.output(task_name_.c_str());
      input_stream.clear();
      input_stream.seekg(0);
    } while (!stop_flag);
    // 训练完成后再最后跑一遍测试集
    while (std::getline(input_stream, line_buff)) {
      solver->feedSample(line_buff.c_str());
      // solver->train_fm_flattern(true);
      solver->train(y, logit, true);
      eval.add(y, logit);
    }
    eval.output(task_name_.c_str());
  }
  ISolver * solver;
};

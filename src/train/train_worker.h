/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <atomic>
#include <thread>

#include "solver/base_solver.h"
#include "train/evalution.h"
#include "utils/busy_consumer_queue.h"

class TrainWorker {
 public:
  TrainWorker(const string & task_name, int task_id)
      : task_name_(task_name),
        task_id_(task_id),
        solver(NULL),
        task_queue(train_opt.task_queue_size),
        stop_flag(false) {}
  ~TrainWorker() {
    if (solver) {
      delete solver;
    }
  }

  void RegisteSolver(BaseSolver *_solver) {
    if (solver) {
      delete solver;
    }
    solver = _solver;
  }

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

  void CollectEvalInfo(Evalution &collect_to) { collect_to += eval; }

 private:
  void task_process_thread_loop() {
    std::vector<string> local_task_queue;
    int counter_for_evalution = 0;
    const bool verbose_debug = train_opt.verbose > 1;

    sleep(2);

    int y;
    real_t logit, loss, grad;
    do {
      task_queue.FeachAll(local_task_queue);
      int task_queue_size = local_task_queue.size();
      if (verbose_debug)
        std::cout << "task " << task_name_ << " fetched " << task_queue_size
                  << " lines" << std::endl;

      for (int i = 0; i < task_queue_size; i++) {
        // train_fm_flattern(y, logit);
        solver->train(local_task_queue[i], y, logit, loss, grad);
        eval.add(y, logit, loss, grad);
      }

      local_task_queue.clear();
      counter_for_evalution += task_queue_size;
      if (counter_for_evalution >= train_opt.n_sample_per_output) {
        counter_for_evalution = 0;
        eval.output(task_name_.c_str());
      }
    } while (!stop_flag);
  }

  void validation_thread_loop(std::ifstream &input_stream) {
    string line_buff;
    int sleep_seconds = 0;
    const bool verbose_debug = train_opt.verbose > 1;
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
        std::cout << "validation thread begin forward... " << std::endl;

      while (std::getline(input_stream, line_buff)) {
        // solver->train_fm_flattern(y, logit, true);
        real_t logit;
        int y;
        solver->test(line_buff, y, logit);
        eval.add(y, logit, 0.0, 0.0);
      }
      eval.output(task_name_.c_str(), true);
      eval.reset();
      input_stream.clear();
      input_stream.seekg(0);
    } while (!stop_flag);
    // do validation after tranning finished
    while (std::getline(input_stream, line_buff)) {
      // solver->train_fm_flattern(true);
      real_t logit;
      int y;
      solver->test(line_buff, y, logit);
      eval.add(y, logit, 0.0, 0.0);
    }
    eval.output(task_name_.c_str(), true);
  }

 private:
  string task_name_;
  int task_id_;
  BaseSolver *solver;
  Evalution eval;
  utils::BusyConsumerQueue<string> task_queue;
  std::thread thread_handler;
  std::atomic<bool> stop_flag;
};

/**
 *  Copyright (c) 2021 by exFM Contributors
 */
#pragma once
#include <condition_variable>
#include <memory>
#include <mutex>
#include <vector>
namespace utils {

template <typename T>
class BusyConsumerQueue {
 public:
  BusyConsumerQueue(size_t full_size = 8000) : full_size_(full_size) {}
  ~BusyConsumerQueue() {}

  bool TryPush(T new_value) {
    bool ret = false;
    if (mu_.try_lock()) {
      if (tasks.size() < full_size_) {
        tasks.push_back(std::move(new_value));
        ret = true;
      }
      mu_.unlock();
    }
    return ret;
  }

  // push when the task is not full
  void WaitAndPush(T new_value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this] { return tasks.size() < full_size_; });
    tasks.push_back(std::move(new_value));
  }

  void FeachAll(std::vector<T>& feach_to) {
    mu_.lock();
    feach_to.swap(tasks);
    mu_.unlock();
    cond_.notify_all();
  }

 private:
  size_t full_size_;
  mutable std::mutex mu_;
  std::vector<T> tasks;
  std::condition_variable cond_;
};

}

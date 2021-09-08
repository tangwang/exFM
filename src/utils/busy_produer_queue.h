/**
 *  copy from Difacto : https://github.com/dmlc/difacto
 */
#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
namespace utils {

/**
 * \brief thread-safe queue allowing push and waited pop
 */
template<typename T> class ThreadsafeQueue {
 public:
  ThreadsafeQueue() { }
  ~ThreadsafeQueue() { }

  /**
   * \brief push an value into the end. threadsafe.
   * \param new_value the value
   */
  void Push(T new_value) {
    mu_.lock();
    queue_.push(std::move(new_value));
    mu_.unlock();
    cond_.notify_all();
  }

  /**
   * \brief wait until pop an element from the beginning, threadsafe
   * \param value the poped value
   */
  void WaitAndPop(T* value) {
    std::unique_lock<std::mutex> lk(mu_);
    cond_.wait(lk, [this]{return !queue_.empty();});
    *value = std::move(queue_.front());
    queue_.pop();
  }

  bool TryPop(T& value) {
    std::lock_guard<std::mutex> lk(mu_);
    if(queue_.empty())
      return false;
    value=std::move(queue_.front());
    queue_.pop();
    return true;
  }

  bool Empty() const {
    std::lock_guard<std::mutex> lk(mu_);
    return queue_.empty();
}

 private:
  mutable std::mutex mu_;
  std::queue<T> queue_;
  std::condition_variable cond_;
};

} 

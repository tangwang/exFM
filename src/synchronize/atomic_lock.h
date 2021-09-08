#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <atomic>

// c++11支持的atomic。
// std::atomic_flag可用于多线程之间的同步操作，类似于linux中的信号量。使用atomic_flag可实现mutex
class AtomicLock {
 protected:
  std::atomic_flag lock_;

 public:
  AtomicLock() { lock_ = ATOMIC_FLAG_INIT; }

  inline void lock() {
    while (lock_.test_and_set())
      ;
  }
  inline int tryLock() {
    lock();
    return 1;
  }

  inline void unlock() { lock_.clear(); }

  inline void wait() {}

  inline void notify() {}
};

#endif

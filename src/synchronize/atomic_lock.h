#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#if 0
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
#else
class AtomicflagSpinLock {
public:
    AtomicflagSpinLock() { m_lock.clear(); }
    //AtomicflagSpinLock(const AtomicflagSpinLock&) = delete;
    //~AtomicflagSpinLock() 

    void lock() {
        while (m_lock.test_and_set(std::memory_order_acquire));
    }
    bool try_lock() {
        return !m_lock.test_and_set(std::memory_order_acquire);
    }
    void unlock() {
        m_lock.clear(std::memory_order_release);
    }
    inline void wait() {}

    inline void notify() {}
private:
    std::atomic_flag m_lock;
};
#endif

#endif

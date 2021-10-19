#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <atomic>
#include <thread>


class AtomicflagSpinLock {
public:
    AtomicflagSpinLock() : m_lock(ATOMIC_FLAG_INIT) {  }

    void lock() {
        while (m_lock.test_and_set(std::memory_order_acquire));
    }
    bool try_lock() {
        return !m_lock.test_and_set(std::memory_order_acquire);
    }
    void unlock() {
        m_lock.clear(std::memory_order_release);
    }
    void wait() {}

    void notify() {}
private:
    std::atomic_flag m_lock;
  // disable copy
  private:
  AtomicflagSpinLock(const AtomicflagSpinLock &ohter);
  AtomicflagSpinLock &operator=(const AtomicflagSpinLock &that);
  AtomicflagSpinLock(AtomicflagSpinLock &ohter);
  AtomicflagSpinLock &operator=(AtomicflagSpinLock &that);
};


// c++11支持的atomic。
// std::atomic_flag可用于多线程之间的同步操作，类似于linux中的信号量。使用atomic_flag可实现mutex
class AtomicSpinLockWithoutMemOder {
 protected:
  std::atomic_flag lock_;

 public:
  AtomicSpinLockWithoutMemOder() :lock_(ATOMIC_FLAG_INIT) {}

  void lock() {
    while (lock_.test_and_set())
      ;
  }
  int tryLock() {
    lock();
    return 1;
  }

  void unlock() { lock_.clear(); }

  void wait() {}

  void notify() {}
  // disable copy
  private:
  AtomicSpinLockWithoutMemOder(const AtomicSpinLockWithoutMemOder &ohter);
  AtomicSpinLockWithoutMemOder &operator=(const AtomicSpinLockWithoutMemOder &that);
};

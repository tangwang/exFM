#pragma once

#include <stdio.h>
#include <stdlib.h>

class GccSpinLock {
 public:
  GccSpinLock() { lock_ = 0; }

  void lock() {
    while (!(__sync_bool_compare_and_swap(&(lock_), 0, 1)))
      ;
  }

  void unlock() { __sync_lock_release(&lock_); }

  int tryLock() {
    return (__sync_bool_compare_and_swap(&(lock_), 0, 1) ? 0 : -1);
  }

  void wait() {}

  void notify() {}
  
 protected:
  volatile unsigned char lock_;
};

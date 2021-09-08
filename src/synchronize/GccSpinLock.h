#ifndef GccSpinLock_H_
#define GccSpinLock_H_

#include <stdio.h>
#include <stdlib.h>

class GccSpinLock {
 protected:
  volatile unsigned char lock_;

 public:
  GccSpinLock() { lock_ = 0; }

  inline void lock() {
    while (!(__sync_bool_compare_and_swap(&(lock_), 0, 1)))
      ;
  }

  inline void unlock() { __sync_lock_release(&lock_); }

  inline int tryLock() {
    return (__sync_bool_compare_and_swap(&(lock_), 0, 1) ? 0 : -1);
  }

  inline void wait() {}

  inline void notify() {}
};

#endif

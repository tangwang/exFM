#pragma once

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

class PthreadMutexWithCond {
 protected:
  pthread_mutex_t lock_;
  pthread_cond_t cond_;

 public:
  PthreadMutexWithCond() {
    pthread_mutex_init(&lock_, (pthread_mutexattr_t *)NULL);
    pthread_cond_init(&cond_, NULL);
  }

  void lock() { pthread_mutex_lock(&lock_); }

  void unlock() { pthread_mutex_unlock(&lock_); }

  int tryLock() { return pthread_mutex_trylock(&lock_); }

  void wait() { pthread_cond_wait(&cond_, &lock_); }

  void notify() { pthread_cond_signal(&cond_); }

  void readLock() { lock(); }

  int tryReadlock() { return tryLock();}

  void writeLock() { lock(); }

  int tryWritelock() { return tryLock(); }

  // disable copy
 private:
  PthreadMutexWithCond(const PthreadMutexWithCond &ohter);
  PthreadMutexWithCond &operator=(const PthreadMutexWithCond &that);
  PthreadMutexWithCond(PthreadMutexWithCond &ohter);
  PthreadMutexWithCond &operator=(PthreadMutexWithCond &that);
};

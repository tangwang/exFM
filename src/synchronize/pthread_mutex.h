#pragma once

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

class PthreadMutex {
 protected:
  pthread_mutex_t lock_;

 public:
  PthreadMutex() { pthread_mutex_init(&lock_, (pthread_mutexattr_t *)NULL); }
  ~PthreadMutex() { pthread_mutex_destroy(&lock_); }

  void lock() { pthread_mutex_lock(&lock_); }

  void unlock() { pthread_mutex_unlock(&lock_); }

  int tryLock() { return pthread_mutex_trylock(&lock_); }

  void readLock() { lock(); }

  int tryReadlock() { lock(); return true;}

  void writeLock() { lock(); }

  int tryWritelock() { lock(); return true; }

  void wait() {}

  void notify() {}

  // disable copy
 private:
  PthreadMutex(const PthreadMutex &ohter);
  PthreadMutex &operator=(const PthreadMutex &that);
  PthreadMutex(PthreadMutex &ohter);
  PthreadMutex &operator=(PthreadMutex &that);
};


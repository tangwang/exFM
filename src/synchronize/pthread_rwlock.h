#pragma once

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

class PthreadRWLock {
 protected:
  pthread_rwlock_t lock_;

 public:
  PthreadRWLock() { pthread_rwlock_init(&lock_, NULL); }

  ~PthreadRWLock() { pthread_rwlock_destroy(&lock_); }

  void readLock() { pthread_rwlock_rdlock(&lock_); }

  int tryReadlock() { return pthread_rwlock_tryrdlock(&lock_); }

  void writeLock() { pthread_rwlock_wrlock(&lock_); }

  int tryWritelock() { return pthread_rwlock_trywrlock(&lock_); }

  void unlock() { pthread_rwlock_unlock(&lock_); }

  // disable copy
 private:
  PthreadRWLock(const PthreadRWLock &ohter);
  PthreadRWLock &operator=(const PthreadRWLock &that);
  PthreadRWLock(PthreadRWLock &ohter);
  PthreadRWLock &operator=(PthreadRWLock &that);
};


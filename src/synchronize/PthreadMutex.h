#ifndef PTHREADMUTEX_H_
#define PTHREADMUTEX_H_

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

class PthreadMutex {
 protected:
  pthread_mutex_t lock_;

 public:
  PthreadMutex() { pthread_mutex_init(&lock_, (pthread_mutexattr_t*)NULL); }
  ~PthreadMutex() { pthread_mutex_destroy(&lock_); }

  inline void lock() { pthread_mutex_lock(&lock_); }

  inline void unlock() { pthread_mutex_unlock(&lock_); }

  inline int tryLock() { return pthread_mutex_trylock(&lock_); }

  inline void wait() {}

  inline void notify() {}

  // 禁用拷贝
  private:
  PthreadMutex(const PthreadMutex &ohter);
  PthreadMutex &operator=(const PthreadMutex &that);
};

#endif /* PTHREADMUTEX_H_ */

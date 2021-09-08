#ifndef NullMutex_H_
#define NullMutex_H_

#include <stdio.h>
#include <stdlib.h>

class NullMutex {
 public:
  inline void lock() {}

  inline void unlock() {}

  inline int tryLock() { return 0; }
  inline void wait() {}

  inline void notify() {}
};

#endif

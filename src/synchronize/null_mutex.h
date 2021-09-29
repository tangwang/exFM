#pragma once

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


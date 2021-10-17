#pragma once

#include <stdio.h>
#include <stdlib.h>

class NullMutex {
 public:
  void lock() {}

  void unlock() {}

  int tryLock() { return 0; }
  
  void wait() {}

  void notify() {}
};


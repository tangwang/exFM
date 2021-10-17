#pragma once

class NullRwMutex {
 public:
  NullRwMutex() {}

  ~NullRwMutex() {}

  void readLock() {}

  int tryReadlock() { return 0; }

  void writeLock() {}

  int tryWritelock() { return 0; }

  void unlock() {}
};

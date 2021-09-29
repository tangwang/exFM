#include "synchronize/gcc_spin_lock.h"
#include "synchronize/pthread_mutex.h"
#include "utils/base.h"

using namespace std;
// condition_variable example
#include <condition_variable>  // std::condition_variable
#include <iostream>            // std::cout
#include <mutex>               // std::mutex, std::unique_lock
#include <string>              //
#include <thread>              // std::thread

// for splin lock
#include <errno.h>
#include <linux/unistd.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#include <list>

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void print_id(int id) {
  std::unique_lock<std::mutex> lck(mtx);
  while (!ready) cv.wait(lck);
  // ...
  std::cout << "thread " << id << '\n';
}

void go() {
  std::unique_lock<std::mutex> lck(mtx);
  ready = true;
  cv.notify_all();
}

int main() {
  std::cout << " sizeof PthreadMutex  " << sizeof(PthreadMutex) << endl;
  std::cout << " sizeof GccSpinLock  " << sizeof(GccSpinLock) << endl;
  std::cout << " sizeof std::mutex  " << sizeof(std::mutex) << endl;
  std::cout << " sizeof std::condition_variable  "
            << sizeof(std::condition_variable) << endl;
  std::cout << " sizeof pthread_spinlock_t  " << sizeof(pthread_spinlock_t)
            << endl;

  std::thread threads[10];
  // spawn 10 threads:
  for (int i = 0; i < 10; ++i) threads[i] = std::thread(print_id, i);

  std::cout << "10 threads ready to race..." << endl;
  go();

  for (auto& th : threads) th.join();

  return 0;
}

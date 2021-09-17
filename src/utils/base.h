#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h> // for sleep
#include <algorithm>
#include <utility>
#include <vector>
#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <iomanip>  // std::setw()
#include <memory>


using std::string;
using std::vector;
using std::pair;
using std::cout;
using std::cin;
using std::cerr;
using std::endl;
using std::shared_ptr;

using std::ifstream;
using std::ofstream;
using std::fstream;

#define WINDOWS_VER_ 0 // 暂时未测试

#define _DEBUG_VER_

#ifdef _DEBUG_VER_
#include <cassert>
#else
#ifdef assert
#undef assert
#endif
#define assert(test) (void(0))
#endif


#ifndef likely
#define likely(x)   __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define UNLIKELY unlikely


// typedef float real_t;
typedef double real_t;

// typedef long feaid_t;
typedef int feaid_t;


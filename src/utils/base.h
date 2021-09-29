#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>  // for sleep
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>  // std::setw()
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::fstream;
using std::ifstream;
using std::istream;
using std::ofstream;
using std::ostream;
using std::pair;
using std::shared_ptr;
using std::string;
using std::vector;
using std::map;

#ifndef DIM
#define DIM 15
#endif
#if !DIM
#define DIM 15
#endif

#define WINDOWS_VER_ 0  // 暂未支持windows版本

#ifdef _DEBUG_VER_
#include <cassert>
#define DEBUG_OUT std::cout
#else
#define DEBUG_OUT NULL&& std::cout
#ifdef assert
#undef assert
#endif
#define assert(test) (void(0))
#endif

#ifdef uint
#undef uint
#endif
typedef unsigned int uint;

#ifdef uint16
#undef uint16
#endif
typedef unsigned short uint16;

#ifdef uint8
#undef uint8
#endif
typedef unsigned char uint8;

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define UNLIKELY unlikely

// typedef float real_t;
typedef double real_t;

// typedef unsigned long feaid_t;
typedef unsigned int feaid_t;

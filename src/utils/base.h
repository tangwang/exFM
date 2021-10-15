#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h> // strcasecmp ...
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
#include <unordered_map>

// containers
using std::string;
using std::vector;
using std::unordered_map;

using std::pair;
using std::shared_ptr;
using std::make_pair;
using std::make_shared;

// IO
using std::cin;
using std::cout;
using std::cerr;
using std::endl;

// file IO
using std::istream;
using std::ostream;
using std::fstream;
using std::ifstream;
using std::ofstream;

// math
using std::max;
using std::min;
using std::sqrt;
using std::log;
using std::exp;

// exception
using std::exception;

// debugs
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

#ifndef likely // likely and unlikely is attribute keywords in C++20
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

// FM
#ifndef DIM
#define DIM 15
#endif
#if !DIM
#define DIM 15
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

// typedef float real_t;
typedef double real_t;

// typedef unsigned long feat_id_t;
typedef unsigned int feat_id_t;

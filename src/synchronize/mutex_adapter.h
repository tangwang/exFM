#pragma once

#include "synchronize/null_mutex.h"
#include "synchronize/pthread_mutex.h"
#include "synchronize/atomic_lock.h"
#include "synchronize/pthread_rwlock.h"
#include "synchronize/null_rwmutex.h"

#ifdef _PREDICT_VER_
typedef NullMutex Mutex_t;
#else
// typedef AtomicflagSpinLock Mutex_t;
typedef PthreadMutex Mutex_t;
#endif

typedef PthreadRWLock RW_Mutex_t;
// typedef NullRwMutex RW_Mutex_t;

 

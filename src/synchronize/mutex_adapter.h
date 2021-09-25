#ifndef mutex_adapter_H_
#define mutex_adapter_H_

#include "synchronize/NullMutex.h"
#include "synchronize/PthreadMutex.h"
#include "synchronize/atomic_lock.h"

#ifdef _PREDICT_VER_
typedef NullMutex Mutex_t;
#else
typedef AtomicflagSpinLock Mutex_t;
// typedef PthreadMutex Mutex_t;
#endif

#endif /* mutex_adapter_H_ */

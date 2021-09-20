#ifndef mutex_adapter_H_
#define mutex_adapter_H_

#include "synchronize/PthreadMutex.h"
#include "synchronize/GccSpinLock.h"
#include "synchronize/NullMutex.h"
#include "synchronize/atomic_lock.h"

#ifdef _PREDICT_VER_
typedef NullMutex Mutex_t;
#else
typedef PthreadMutex Mutex_t;
// typedef AtomicflagSpinLock Mutex_t;  // TODO check 性能与PthreadMutex对比
#endif

// TODO spinlock是否能保证内存同步。 如果可以的话替换为spinlock，在并发数大于20的时候性能有很大的提升
//typedef GccSpinLock Mutex_t;
// typedef NullMutex Mutex_t;

#endif /* mutex_adapter_H_ */

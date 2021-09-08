#ifndef PTHREADRWLOCK_H_
#define PTHREADRWLOCK_H_


#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

class PthreadRWLock{
protected:
	pthread_rwlock_t lock_;


public:
	PthreadRWLock(){
		pthread_rwlock_init(&lock_, NULL);
	}
	~PthreadRWLock(){
		pthread_rwlock_destroy(&lock_);
	}

	inline void rdlock(){
		pthread_rwlock_rdlock(&lock_);
	}

	inline int tryRdlock(){
		return pthread_rwlock_tryrdlock(&lock_);
	}

	inline void wrlock(){
		pthread_rwlock_wrlock(&lock_);
	}

	inline int tryWrlock(){
		return pthread_rwlock_trywrlock(&lock_);
	}

	inline void unlock(){
		pthread_rwlock_unlock(&lock_);
	}

};





#endif /* PTHREADRWLOCK_H_ */

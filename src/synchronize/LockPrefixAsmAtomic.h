#ifndef LOCKPREFIXASMATOMIC_H_
#define LOCKPREFIXASMATOMIC_H_


#if __GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 1)
/* Technically wrong, but this avoids compilation errors on some gcc
   versions. */
#define ADDR "=m" (*(volatile long *) addr)
#else
#define ADDR "+m" (*(volatile long *) addr)
#endif

#ifdef CONFIG_SMP
#define LOCK_PREFIX "lock; "
#else /* ! CONFIG_SMP */
#define LOCK_PREFIX ""
#endif

typedef struct {
    int counter;
} atomic_t;

#define atomic_read(v)        ((v)->counter)
#define atomic_set(v, i)        (((v)->counter) = (i))

static inline void atomic_inc(atomic_t *v)
{
    asm volatile(LOCK_PREFIX "incl %0"
             : "=m" (v->counter)
             : "m" (v->counter));
}

static inline void atomic_dec(atomic_t *v)
{
    asm volatile(LOCK_PREFIX "decl %0"
             : "=m" (v->counter)
             : "m" (v->counter));
}

static inline void set_bit(int nr, volatile void *addr)
{
    asm volatile(LOCK_PREFIX "bts %1,%0" : ADDR : "Ir" (nr) : "memory");
}

static inline int test_and_set_bit(int nr, volatile void *addr)
{
    int oldbit;

    asm volatile(LOCK_PREFIX "bts %2,%1nt"
             "sbb %0,%0" : "=r" (oldbit), ADDR : "Ir" (nr) : "memory");

    return oldbit;
}

static inline void clear_bit(int nr, volatile void *addr)
{
    asm volatile(LOCK_PREFIX "btr %1,%0" : ADDR : "Ir" (nr));
}

static inline int test_and_clear_bit(int nr, volatile void *addr)
{
    int oldbit;

    asm volatile(LOCK_PREFIX "btr %2,%1nt"
             "sbb %0,%0"
             : "=r" (oldbit), ADDR : "Ir" (nr) : "memory");

    return oldbit;
}


#endif /* LOCKPREFIXASMATOMIC_H_ */

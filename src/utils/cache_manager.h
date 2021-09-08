#ifndef CACHE_MANAGER_H
#define CACHE_MANAGER_H

#include <map>
#include <string>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <pthread.h>
#include "util/gtool.h"
#include "util/List.h"
#include "util/utils/Hash.h"

namespace utils {

#define SHM_PERM 0600
#define SEM_PERM 0600

/*
 * 共享内存结构：
 * |              |             索引区                   |           数据节点区               |
 * |              |<---- hash_buckets_num ---->|<----- max_cache_num ----->|
 * |______________|____________________________|___________________________|
 * | ShareMemHead |HashEntry ...  ... HashEntry|DataNode  ... ...  DataNode|
 * |______________|____________________________|___________________________|
 *
 */

typedef unsigned int offset_type;

struct CacheConfig {
    std::string dir;
    int switch_;
    int sequence;
    int hash_buckets_num;
    int max_cache_num;
    int ksize;
    int vsize;

    CacheConfig() :
    switch_(0),
    sequence(0),
    hash_buckets_num(0),
    max_cache_num(0),
    ksize(0),
    vsize(0)
    {}

};

struct HashEntry
{
    HashEntry() {
        init();
    }
    inline void init() {
        list_head_init(&bucketHead);
    }
    struct list_head bucketHead;
};

class LocalCache {

public:
    LocalCache();
    ~LocalCache();

    bool open(const CacheConfig & param);
    bool attach();
    void detach();
    bool reset_shm();

    bool get(const std::string & k, string & v);
    bool set(const std::string & k, const std::string & value, int lease);

//    bool get(const std::string & k, unsigned char *v, int v_size);
//    bool set(const std::string & k, const unsigned char *v, int v_size, int lease);
//
//    template<typename ValueType>
//    bool get(const std::string & k, ValueType & v) {
//        get(k, &v, sizeof(v));
//    }
//
//    template<typename ValueType>
//    bool set(const std::string & k, const ValueType & v, int lease) {
//        set(k, &v, sizeof(v), lease);
//    }

    std::string debug(const char * msg = "", bool show_detail = false);

private:
    bool cache_ready;
    int shm_id;                      //共享内存标识符
    int sem_id;                      //信号量标识符

    int shmhead_size;
    int node_size;
    int share_mem_size;

    CacheConfig conf;

    struct DataNode {
        struct list_head link;
        struct list_head visitList;
        int expired_time;
        char buff[];

        DataNode():expired_time(0) {
            init();
        }
        inline void init() {
            list_head_init(&link);
            list_head_init(&visitList);
        }
    };

    struct Statis {
        Statis() {
            clear();
        }
        void clear() {
            lock_busy = 0;
            set_total = 0;
            get_total = 0;
            cache_hit = 0;
            cache_miss = 0;
            get_invalid = 0;
            set_invalid = 0;
            lru_eliminated = 0;
        }
        long lock_busy;
        long set_total;
        long get_total;
        long cache_hit;
        long cache_miss;
        long get_invalid;
        long set_invalid;
        long lru_eliminated;
    } statis;

    struct ShareMemHead {
        offset_type idle_position;
        HashEntry idle_list;
        HashEntry visit_list;
        pthread_mutex_t global_lock;
//        pthread_mutex_t hash_entries_lock;
//        pthread_mutex_t visit_list_lock;
//        pthread_mutex_t idle_list_lock;

        Statis statis;
    };

    struct ShmHander {
        unsigned char *base_addr;                 //共享内存区起始地址, 为NULL表示非attach状态
        ShareMemHead * shmem_head; // 头部
        HashEntry * hash_entries;  // 索引区起始地址
        unsigned char *nodes_base_addr; // 数据块起始地址
        unsigned char *nodes_end_addr;  // 数据块终止地址

        void clear() {
            base_addr = NULL;
            shmem_head = NULL;
            hash_entries = NULL;
            nodes_base_addr = NULL;
            nodes_end_addr = NULL;
        }

        ShmHander() {
            clear();
        }
    } shm;


private:
    bool lock_mtx();
    bool trylock_mtx();
    void unlock_mtx();

    bool lock_sem();
    void unlock_sem();

    bool lock();
    bool trylock();
    void unlock();

    void touch(DataNode * node);

    static int get_current_time();
    int get_bucket_idx(const std::string & key);


};

}// namespace utils

#endif /* CACHE_MANAGER_H */

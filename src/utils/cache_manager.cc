#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include "qss_common/logging.h"
#include "util/gtool.h"
#include "module.h"
#include "cache_manager.h"
#include "util/stopwatch.h"

using namespace std;

namespace utils{

LOGGING_INIT("LocalCache");

int LocalCache::get_bucket_idx(const std::string & key) {
    return hsieh_hash32_str(key) % conf.hash_buckets_num;
}

LocalCache::LocalCache():
    cache_ready(false),
    shm_id(-1),
    sem_id(-1),
    shmhead_size(0),
    node_size(0),
    share_mem_size(0)
{
}

LocalCache::~LocalCache() {
    // 只能由一个销毁锁，不能所有attach上去的进程都执行销毁锁
    if (shm.shmem_head) {
        pthread_mutex_destroy(&shm.shmem_head->global_lock);
    }
    detach();
}

bool LocalCache::open(const CacheConfig & cf) {

    conf = cf;
    int min_buff_size = sizeof(DataNode) - offset(DataNode, buff);
    if (conf.ksize + conf.vsize <= min_buff_size) {
        node_size = sizeof(DataNode);
    } else {
        node_size = sizeof(DataNode) - min_buff_size + conf.ksize + conf.vsize;
    }

    while (node_size % sizeof(long) != 0) {
        ++node_size;
    }
    shmhead_size = sizeof(ShareMemHead);
    while (shmhead_size % sizeof(long) != 0) {
        ++shmhead_size;
    }
    share_mem_size = shmhead_size + conf.hash_buckets_num * sizeof(HashEntry) + conf.max_cache_num * node_size;

    //检查sequence合法性
    if(conf.sequence < 1 || conf.sequence > 255) {
        return false;
    }

    key_t shm_key;
    shm_key = ftok(conf.dir.c_str(), conf.sequence);
    shm_id = shmget(shm_key, share_mem_size, SHM_PERM|IPC_CREAT);
    if(shm_id == -1) {
        LOG(ERROR) << " cache_log~~ shmget failed";
        return false;
    }

    key_t sem_key;
    sem_key = ftok(conf.dir.c_str(), conf.sequence);
    sem_id = semget(sem_key, 1, SEM_PERM|IPC_CREAT);
    if(sem_id == -1) {
        LOG(ERROR) << " cache_log~~ semget failed";
        return false;
    }

    //保存shm_id和sem_id
    ofstream ofs;
    string shmid_dir = conf.dir + "../proc/utils.shmid";
    ofs.open(shmid_dir.c_str(), ios::app);
    if(!ofs) {
        return false;
    }
    ofs << shm_id << "\t" << sem_id << "\n";
    ofs.close();
    cache_ready = true;

    struct shmid_ds shm_ds;
    int flag = shmctl(shm_id, IPC_STAT, &shm_ds);
    if (flag == -1) {
        return false;
    }
    if (shm_ds.shm_nattch == 0) {
        LOG(DEBUG) << " cache_log~~ OPEN: first init, reset_shm!";
        attach();
        reset_shm();
    } else {
        attach();
        LOG(DEBUG) << " cache_log~~ OPEN: open only, nattach: " << shm_ds.shm_nattch;
    }

    return true;
}

bool LocalCache::attach() {
    if (!cache_ready) return false;
    if (shm.base_addr) return true;

    shm.base_addr = (unsigned char *)shmat(shm_id, NULL, 0);
    if (shm.base_addr == (unsigned char *)-1) {
        LOG(ERROR) << " cache_log~~ shmat failed";
        return false;
    }
    LOG(DEBUG) << " cache_log~~ shmat success: addr " << (unsigned long)shm.base_addr;

    shm.shmem_head = (ShareMemHead *) shm.base_addr;
    shm.hash_entries = (HashEntry *)(shm.base_addr + shmhead_size);
    shm.nodes_base_addr = (unsigned char *)(shm.hash_entries + conf.hash_buckets_num);
    shm.nodes_end_addr = shm.nodes_base_addr + conf.max_cache_num*node_size;
    return true;
}

bool LocalCache::reset_shm() {
    LOG(DEBUG) << " cache_log~~ reset_shm, addr: " << (unsigned long)shm.base_addr;
    shm.shmem_head->idle_list.init();
    shm.shmem_head->visit_list.init();
    shm.shmem_head->idle_position = shm.nodes_base_addr - shm.base_addr;

    for (int i = 0; i < conf.hash_buckets_num; ++i) {
        shm.hash_entries[i].init();
    }

    pthread_mutexattr_t mattr;
    if (0 != pthread_mutexattr_init(&mattr)
        || 0 != pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_SHARED)
        || 0 != pthread_mutexattr_setrobust(&mattr, PTHREAD_MUTEX_ROBUST)
        || 0 != pthread_mutex_init(&shm.shmem_head->global_lock, &mattr)) {
        LOG(ERROR) << " cache_log~~ pthread_mutex init failed: " << " errno: " << errno << " strerror: " << strerror(errno);
        return false;
    }
    return true;
}

void LocalCache::detach() {
    if (shm.base_addr == NULL) {
        return;
    }
    LOG(DEBUG) << " cache_log~~ shmat detach: addr " << (unsigned long)shm.base_addr;
    if (shmdt(shm.base_addr) == -1) {
        LOG(ERROR) << " cache_log~~ shmdt failed";
    } else {
        LOG(DEBUG) << " cache_log~~ shmdt success";
    }
    shm.clear();
}

bool LocalCache::lock() {
    return lock_mtx();
}
bool LocalCache::trylock() {
    return trylock_mtx();
}
void LocalCache::unlock() {
    unlock_mtx();
}

bool LocalCache::lock_mtx() {
    return 0 == pthread_mutex_lock(&shm.shmem_head->global_lock);
}

bool LocalCache::trylock_mtx() {
    int r = pthread_mutex_trylock(&shm.shmem_head->global_lock);
    if (r != 0) {
        if (r == EOWNERDEAD) {
            pthread_mutex_consistent(&shm.shmem_head->global_lock);
            pthread_mutex_unlock(&shm.shmem_head->global_lock);
            LOG(ERROR) << " cache_log~~ pthread_mutex_consistent!";
        }
        ++ statis.lock_busy;
        LOG(ERROR) << " cache_log~~ lock failed, ret " << r << ", errno: " << errno << " strerror: " << strerror(errno);
        return false;
    }
    return true;
}

void LocalCache::unlock_mtx() {
    pthread_mutex_unlock(&shm.shmem_head->global_lock);
}

bool LocalCache::lock_sem() {
    struct sembuf ops[2];
    ops[0].sem_num = 0;
    ops[0].sem_op = 0;
    ops[0].sem_flg = SEM_UNDO|IPC_NOWAIT;
    ops[1].sem_num = 0;
    ops[1].sem_op = 1;
    ops[1].sem_flg = SEM_UNDO|IPC_NOWAIT;
//    static struct sembuf sem_lock = { 0, -1, SEM_UNDO | IPC_NOWAIT };

    if (semop(sem_id, &ops[0], 2) == -1){
        ++ statis.lock_busy;
        LOG(ERROR) << " cache_log~~ lock semop failed, sem id: " << sem_id << " errno: " << errno << " strerror: " << strerror(errno);
        return false;
    }
    return true;
}

void LocalCache::unlock_sem() {
    struct sembuf ops[1];
    ops[0].sem_num = 0;
    ops[0].sem_op = -1;
    ops[0].sem_flg = SEM_UNDO|IPC_NOWAIT;

//    static struct sembuf sem_unlock = { 0, 1, SEM_UNDO | IPC_NOWAIT };

    if(semop(sem_id, &ops[0], 1) == -1){
        LOG(ERROR) << " cache_log~~ unlock semop failed";
    }
}

bool LocalCache::get(const std::string & k, string & v)
{
    ++ statis.get_total;

    if (shm.base_addr == NULL || k.empty() || k.length() >= conf.ksize) {
//        LOG(DEBUG) << " cache_log~~ get invalid, key: " << k << " length " << k.length();
        ++ statis.get_invalid;
        return false;
    }
    int hashKey = get_bucket_idx(k);

    bool ret = false;
    HashEntry *hashHead;
    DataNode *node;
    DataNode *tempNode;

    hashHead = shm.hash_entries + hashKey;

    if (!trylock()) {
        return false;
    }

    list_for_each_entry_safe(node, tempNode, &hashHead->bucketHead, link)
    {
        if (k != node->buff) continue;

        if (node->expired_time > get_current_time()) {
            v = node->buff + conf.ksize;
            touch(node);
            ret = true;
        }
        break;
    }

    unlock();

    ret ? ++statis.cache_hit : ++statis.cache_miss;
    return ret;
}

bool LocalCache::set(const std::string & k, const std::string & v, int lease) {

//    LOG(DEBUG) << " cache_log~~: set " << k << " : " << v.substr(0, 25);

    ++ statis.set_total;

    if (shm.base_addr == NULL
       || k.empty() || k.length() >= conf.ksize || v.length() >= conf.vsize
       || lease <= 0) {
        ++ statis.set_invalid;
        return false;
    }

    int hashKey = get_bucket_idx(k);
    bool ret = false;
    HashEntry *hashHead;
    DataNode *node;
    DataNode *tempNode;

    hashHead = shm.hash_entries + hashKey;

    if (!trylock()) {
        return false;
    }
    list_for_each_entry_safe(node, tempNode, &hashHead->bucketHead, link)
    {
        if (k != node->buff) continue;
        memcpy(node->buff + conf.ksize, v.c_str(), v.length() + 1);
        ret = true;
        break;
    }

    if (!ret) {
        if (!list_empty(&shm.shmem_head->idle_list.bucketHead)) {
            // 先取空闲队列头
            node = list_entry(shm.shmem_head->idle_list.bucketHead.next, DataNode, link);
            list_del_(&node->link);
        } else if (shm.base_addr + shm.shmem_head->idle_position != shm.nodes_end_addr) {
            // 剩余地址
            node = (DataNode *)(shm.base_addr + shm.shmem_head->idle_position);
            shm.shmem_head->idle_position += node_size;
            node->init();
        } else {
            ++statis.lru_eliminated;
            // 再取lru队列尾
            node = list_entry(shm.shmem_head->visit_list.bucketHead.prev, DataNode, visitList);
            list_del_(&node->link);
        }
        list_add_(&node->link, &hashHead->bucketHead);
        memcpy(node->buff, k.c_str(), k.length() + 1);
    }

    node->expired_time = get_current_time() + lease;
    memcpy(node->buff + conf.ksize, v.c_str(), v.length() + 1);
    touch(node);

    unlock();
    return true;
}

void LocalCache::touch(DataNode * node) {
    list_del_(&node->visitList);
    list_add_(&node->visitList, &shm.shmem_head->visit_list.bucketHead);
}

int LocalCache::get_current_time() {
    time_t now;
    time(&now);
    return (int)now;
}

std::string LocalCache::debug(const char * msg, bool show_detail) {
    string ret;
    ret.reserve(10240);
    if (shm.base_addr == NULL) {
        ret = msg;
        ret += " : shm.base_addr NULL" << endl;
        return ret;
    }
    char buff[1024];
    snprintf(buff, sizeof(buff), "[%s shm_stat] pid %d, conf.hash_buckets_num %d, conf.max_cache_num %d, shm.base_addr %lx, SHARED_MEMORY_SZ %dM, node_size %d, shm_id %d, sem_id %d", msg, getpid(), conf.hash_buckets_num, conf.max_cache_num, (unsigned long)shm.base_addr, share_mem_size/(1024*1024), node_size, shm_id, sem_id);
    LOG(DEBUG) << "  cache_log~~ " << buff;
    ret += buff;
    ret += "\n";
    snprintf(buff, sizeof(buff), "[%s cache_stat] lock_busy %ld, set_total %ld, get_total %ld, cache_hit %ld, cache_miss %ld, get_invalid %ld, set_invalid %ld, lru_eliminated %ld", msg, statis.lock_busy, statis.set_total, statis.get_total, statis.cache_hit, statis.cache_miss, statis.get_invalid, statis.set_invalid, statis.lru_eliminated);
    LOG(DEBUG) << "  cache_log~~ " << buff;
    ret += buff;
    ret += "\n";

    if (!show_detail) {
        return ret;
    }

    if (!trylock()) {
        ret += " lock failed!!!" << endl;
        return ret;
    }

    DataNode *node;
    DataNode *tempNode;
    char v[50] = {0};

    // 最近20条
    ret += "latest 20 entries:" << endl;
    int max_count = 20;
    list_for_each_entry_safe(node, tempNode, &shm.shmem_head->visit_list.bucketHead, visitList)
    {
        if (--max_count == 0) break;

        strncpy(v, node->buff + conf.ksize, sizeof(v)-1);
        ret += node->buff;
        ret += "    ->    ";
        ret += v;
        ret += "\n";
    }
    ret += "\n";

    unlock();

    return ret;
}

} // namespace utils

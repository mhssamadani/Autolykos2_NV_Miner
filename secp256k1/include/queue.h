#ifndef QUEUE_H
#define QUEUE_H
#include "definitions.h"
#include <mutex>
#include <condition_variable>
#include <vector>
#include <deque>
#include <iostream>
// all that we need to send to node/pool
struct MinerShare
{
    MinerShare();
    MinerShare(uint64_t _nonce)
    {
        nonce = _nonce;
    }
    uint64_t nonce;
};

//simple blocking queue for solutions sending
template<class T> class BlockQueue
{
    std::deque<T> cont;
    std::mutex mut;
    std::condition_variable condv;
public:    
    void put(T &val)
    {
        mut.lock();
        cont.push_front(val);
        mut.unlock();
        condv.notify_one();

    }
    
    void put(T &&val)
    {
        mut.lock();
        cont.push_front(val);
        mut.unlock();
        condv.notify_one();

    }

    T get()
    {
        std::unique_lock<std::mutex> lock(mut);
        condv.wait(lock, [=]{ 
            return !cont.empty(); });
        T tmp = cont.back();
        cont.pop_back();
        return tmp;
    }
};



#endif

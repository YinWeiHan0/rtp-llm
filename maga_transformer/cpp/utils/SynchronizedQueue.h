#pragma once

#include <deque>
#include <stdint.h>

#include "maga_transformer/cpp/utils/Lock.h"

namespace rtp_llm {

template <class T>
class SynchronizedQueue {
public:
    SynchronizedQueue();
    ~SynchronizedQueue();

public:
    uint32_t getCapacity() const;
    void setCapacity(uint32_t);
    void push(const T &element);
    void push(T &&element);
    bool tryPush(const T &element);
    T getFront();
    void popFront();
    T getAndPopFront();
    bool tryGetAndPopFront(T &element);
    bool isEmpty();
    void waitNotEmpty();
    uint32_t getSize();
    void wakeup();
    void clear();
    void swap(SynchronizedQueue<T> &rhs);

public:
    static const uint32_t DEF_WAIT_TIME = 1000000; // 1 second
    static const uint32_t DEF_CAPACITY = 100000;

private:
    uint32_t _capacity;
    std::deque<T> _elements;
    ThreadCond _cond;
};

template <class T>
SynchronizedQueue<T>::SynchronizedQueue() : _capacity(DEF_CAPACITY) {}

template <class T>
SynchronizedQueue<T>::~SynchronizedQueue() {}

template <class T>
uint32_t SynchronizedQueue<T>::getCapacity() const {
    return _capacity;
}

template <class T>
void SynchronizedQueue<T>::setCapacity(uint32_t capacity) {
    _capacity = capacity;
}

template <class T>
void SynchronizedQueue<T>::push(const T &element) {
    ScopedLock lk(_cond);
    while (_elements.size() >= _capacity) {
        _cond.wait(DEF_WAIT_TIME);
    }
    _elements.push_back(element);
    _cond.signal();
}

template <class T>
void SynchronizedQueue<T>::push(T &&element) {
    ScopedLock lk(_cond);
    while (_elements.size() >= _capacity) {
        _cond.wait(DEF_WAIT_TIME);
    }
    _elements.push_back(std::move(element));
    _cond.signal();
}

template <class T>
bool SynchronizedQueue<T>::tryPush(const T &element) {
    ScopedLock lk(_cond);
    if (_elements.size() >= _capacity) {
        return false;
    }
    _elements.push_back(element);
    _cond.signal();
    return true;
}

template <class T>
T SynchronizedQueue<T>::getFront() {
    ScopedLock lk(_cond);
    const T &element = _elements.front();
    return element;
}

template <class T>
void SynchronizedQueue<T>::popFront() {
    ScopedLock lk(_cond);
    _elements.pop_front();
    _cond.signal();
}

template <class T>
T SynchronizedQueue<T>::getAndPopFront() {
    ScopedLock lk(_cond);

    const T element = _elements.front();
    _elements.pop_front();
    _cond.signal();

    return element;
}

template <class T>
bool SynchronizedQueue<T>::tryGetAndPopFront(T &element) {
    ScopedLock lk(_cond);

    if (_elements.empty()) {
        return false;
    } else {
        element = _elements.front();
        _elements.pop_front();
        _cond.signal();

        return true;
    }
}

template <class T>
void SynchronizedQueue<T>::waitNotEmpty() {
    ScopedLock lk(_cond);
    _cond.wait(DEF_WAIT_TIME);
}

template <class T>
bool SynchronizedQueue<T>::isEmpty() {
    ScopedLock lk(_cond);
    return _elements.empty();
}

template <class T>
uint32_t SynchronizedQueue<T>::getSize() {
    ScopedLock lk(_cond);
    return _elements.size();
}

template <class T>
void SynchronizedQueue<T>::wakeup() {
    ScopedLock lk(_cond);
    _cond.broadcast();
}

template <class T>
void SynchronizedQueue<T>::clear() {
    ScopedLock lk(_cond);
    _elements.clear();
}

template <class T>
void SynchronizedQueue<T>::swap(SynchronizedQueue<T> &rhs) {
    ScopedLock lk(_cond);
    _elements.swap(rhs._elements);
    _cond.broadcast();
}

}  // namespace rtp_llm


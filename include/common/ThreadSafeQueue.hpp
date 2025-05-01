#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class ThreadSafeQueue {
public:
    ThreadSafeQueue() : is_shutdown_(false) {}

    // 添加元素
    void push(const T& value) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (is_shutdown_) return; // 队列关闭后不再添加
            queue_.push(value);
        }
        cond_.notify_one();
    }

    // 阻塞等待直到弹出一个元素，返回 false 表示已关闭
    bool wait_and_pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]() {
            return is_shutdown_ || !queue_.empty();
        });

        if (is_shutdown_ && queue_.empty()) {
            return false;
        }

        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // 尝试弹出一个元素（非阻塞）
    bool try_pop(T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    // 获取当前队列大小
    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    // 判断队列是否为空
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    // 安全关闭队列
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            is_shutdown_ = true;
        }
        cond_.notify_all();
    }

    // 查询是否关闭
    bool is_shutdown() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return is_shutdown_;
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cond_;
    bool is_shutdown_;
};

#ifndef ARMOR_DETECTOR_OPENVINO__THREADPOOL_H
#define ARMOR_DETECTOR_OPENVINO__THREADPOOL_H

#pragma once

#include <vector>
#include <deque>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <iostream>
#include <chrono>
#include <queue>
#include "common/ThreadSafeQueue.hpp"

class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads, size_t max_pending_tasks = 100, int max_task_duration_ms = 100)
        : stop_(false), max_pending_tasks_(max_pending_tasks), max_task_duration_ms_(max_task_duration_ms)
    {
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back([this]() { this->workerThread(); });
        }

        controller_ = std::thread([this]() { this->controlThread(); });
    }

    ~ThreadPool() {
        waitUntilEmpty();  // 等待任务处理完毕

        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }

        cond_var_.notify_all();

        if (controller_.joinable()) {
            controller_.join();
        }

        for (std::thread &worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    template<class F>
    void enqueue(F&& f, bool high_priority = false) {
        {
            std::unique_lock<std::mutex> lock(mutex_);

            if (tasks_.size() >= max_pending_tasks_) {
                tasks_.pop_front();
                std::cerr << "[ThreadPool] Warning: Dropped oldest pending task\n";
            }

            TaskItem task;
            task.func = std::forward<F>(f);
            task.high_priority = high_priority;

            if (high_priority) {
                tasks_.emplace_front(std::move(task));
            } else {
                tasks_.emplace_back(std::move(task));
            }
        }

        cond_var_.notify_one();
    }
    

    size_t pendingTasks() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return tasks_.size();
    }

    void waitUntilEmpty() {
        std::unique_lock<std::mutex> lock(mutex_);
        task_done_cv_.wait(lock, [this]() {
            return tasks_.empty() && busy_workers_ == 0;
        });
    }

private:
    struct TaskItem {
        std::function<void()> func;
        bool high_priority;
    };

    void workerThread() {
        while (true) {
            TaskItem task;

            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_var_.wait(lock, [this]() { return this->stop_ || !this->tasks_.empty(); });

                if (stop_ && tasks_.empty())
                    return;

                task = std::move(tasks_.front());
                tasks_.pop_front();
                ++busy_workers_;
            }

            auto start_time = std::chrono::steady_clock::now();
            task.func();
            auto end_time = std::chrono::steady_clock::now();

            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
            if (duration > max_task_duration_ms_) {
                std::cerr << "[ThreadPool] Warning: Task took too long: " << duration << " ms\n";
            }

            {
                std::unique_lock<std::mutex> lock(mutex_);
                --busy_workers_;
                if (tasks_.empty() && busy_workers_ == 0) {
                    task_done_cv_.notify_all();
                }
            }
        }
    }

    void controlThread() {
        using namespace std::chrono_literals;

        while (!stop_) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                size_t pending = tasks_.size();

                if (pending > max_pending_tasks_ * 0.8) {
                    max_pending_tasks_ = std::max<size_t>(10, max_pending_tasks_ * 0.8);
                    std::cerr << "[ThreadPool] Warning: Queue overloaded, shrink to "
                              << max_pending_tasks_ << "\n";
                } else if (pending < max_pending_tasks_ * 0.3) {
                    max_pending_tasks_ = std::min<size_t>(500, max_pending_tasks_ + 5);
                }
            }
            std::this_thread::sleep_for(500ms);
        }
    }

    std::vector<std::thread> workers_;
    std::deque<TaskItem> tasks_;
    mutable std::mutex mutex_;
    std::condition_variable cond_var_;
    std::condition_variable task_done_cv_;
    std::atomic<size_t> busy_workers_{0};
    bool stop_;
    size_t max_pending_tasks_;
    int max_task_duration_ms_;
    std::thread controller_;
};
inline void SetRealtimePriority(int priority = 90) {
    pthread_t this_thread = pthread_self();
    struct sched_param schedParams;
    schedParams.sched_priority = priority;

    int ret = pthread_setschedparam(this_thread, SCHED_FIFO, &schedParams);
    if (ret != 0) {
        std::cerr << "Failed to set real-time priority. Error code: " << ret << std::endl;
        perror("pthread_setschedparam");
    } else {
        std::cout << "Real-time priority set successfully to " << priority << std::endl;
    }
}
#endif  // ARMOR_DETECTOR_OPENVINO__THREADPOOL_H

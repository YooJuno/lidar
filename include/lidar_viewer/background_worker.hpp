#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <utility>

namespace lidar_viewer
{
// One persistent thread, one running job and at most one pending job. Replacing
// a pending task cancels its future; destroying a future never waits for I/O.
class BackgroundWorker {
public:
  BackgroundWorker():thread_([this]{Run();}) {}
  ~BackgroundWorker() {
    {std::lock_guard<std::mutex> lock(mutex_);stop_=true;pending_={};}
    ready_.notify_one();thread_.join();
  }
  BackgroundWorker(const BackgroundWorker &)=delete;
  BackgroundWorker & operator=(const BackgroundWorker &)=delete;

  template<class F> auto Submit(F && function) -> std::future<std::invoke_result_t<F>> {
    auto task=std::make_shared<std::packaged_task<std::invoke_result_t<F>()>>(std::forward<F>(function));
    auto future=task->get_future();
    {std::lock_guard<std::mutex> lock(mutex_);pending_=[task]{(*task)();};}
    ready_.notify_one();return future;
  }
  void CancelPending() {std::lock_guard<std::mutex> lock(mutex_);pending_={};}

private:
  void Run() {
    for (;;) {
      std::function<void()> task;
      {std::unique_lock<std::mutex> lock(mutex_);ready_.wait(lock,[this]{return stop_ || pending_;});
        if (stop_) return;
        task=std::move(pending_);pending_={};}
      task();
    }
  }
  std::mutex mutex_;
  std::condition_variable ready_;
  bool stop_{false};
  std::function<void()> pending_;
  std::thread thread_;
};
} // namespace lidar_viewer

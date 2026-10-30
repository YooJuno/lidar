#include "lidar_viewer/background_worker.hpp"
#include <chrono>
#include <iostream>
#include <stdexcept>

namespace {
void Check(bool value,const char * message) {if (!value) throw std::runtime_error(message);}
}
int main() {
  try {
    using namespace std::chrono;
    lidar_viewer::BackgroundWorker worker;
    std::promise<void> entered,release;
    auto gate=release.get_future().share();
    auto first=worker.Submit([&]{entered.set_value();gate.wait();return std::this_thread::get_id();});
    entered.get_future().wait();
    auto replaced=worker.Submit([]{throw std::runtime_error("Superseded job ran");return 0;});
    auto last=worker.Submit([]{return std::this_thread::get_id();});
    Check(replaced.wait_for(seconds(0))==std::future_status::ready,"Superseded future was not canceled");
    bool canceled=false;
    try {replaced.get();} catch (const std::future_error &) {canceled=true;}
    Check(canceled,"Replaced job retained its captured memory");
    const auto start=steady_clock::now();
    first={}; // No std::async destructor wait while the running job is blocked.
    Check(steady_clock::now()-start<milliseconds(50),"Discarding a read blocked the caller");
    release.set_value();const auto id=last.get();
    Check(id!=std::this_thread::get_id(),"Heavy job ran on the UI thread");
    for (int i=0;i<500;++i)
      Check(worker.Submit([]{return std::this_thread::get_id();}).get()==id,"Worker thread was recreated per frame");
    bool propagated=false;
    try {worker.Submit([]{throw std::runtime_error("fixture error");}).get();}
    catch (const std::runtime_error &) {propagated=true;}
    Check(propagated,"Background exception was lost");
    Check(worker.Submit([]{return 7;}).get()==7,"A failed job killed the reusable worker");
    std::cout<<"background_worker_smoke: bounded queue, cancellation, nonblocking discard, reuse and exceptions passed\n";
    return 0;
  } catch (const std::exception & error) {std::cerr<<error.what()<<'\n';return 1;}
}

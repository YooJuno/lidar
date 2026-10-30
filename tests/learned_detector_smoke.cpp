#include "lidar_viewer/learned_detector.hpp"
#include "lidar_viewer/platform.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <thread>

using namespace lidar_viewer;
namespace {
void Check(bool condition,const char * message) {if (!condition) throw std::runtime_error(message);}
void ConfigEnvironment(const std::string & value) {
#ifdef _WIN32
  _putenv_s("XDG_CONFIG_HOME",value.c_str());
#else
  if (value.empty()) unsetenv("XDG_CONFIG_HOME");else setenv("XDG_CONFIG_HOME",value.c_str(),1);
#endif
}
template<class F> void Wait(F ready) {
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(15);
  while (!ready()) {Check(std::chrono::steady_clock::now()<deadline,"Detector test timed out");std::this_thread::sleep_for(std::chrono::milliseconds(5));}
}
}
int main(int argc,char ** argv) {
  const std::string previous=std::getenv("XDG_CONFIG_HOME") ? std::getenv("XDG_CONFIG_HOME") : "";
  std::filesystem::path temporary;
  try {
    Check(argc==2,"Pass Python interpreter");temporary=TemporaryDirectory("lidar-detector-test-")/std::filesystem::u8path("runtime space 한글");
    std::filesystem::create_directories(temporary);ConfigEnvironment(temporary.u8string());
    const auto folder=UserConfigurationDirectory();std::filesystem::create_directories(folder);
    const auto worker=temporary/"detector_worker.py";
    std::ofstream(worker)<<R"PY(import struct,sys,time
source=sys.stdin.buffer;sink=sys.stdout.buffer
req=struct.Struct('<4sQIf');reply=struct.Struct('<4sQIfI')
def exact(n):
 b=b''
 while len(b)<n:
  c=source.read(n-len(b))
  if not c:raise SystemExit()
  b+=c
 return b
def send(b):
 for i in range(0,len(b),3):sink.write(b[i:i+3]);sink.flush()
send(reply.pack(b'LDR1',0,0,0,4)+b'CPU!')
while True:
 tag,revision,count,score=req.unpack(exact(req.size));data=exact(count*12)
 if tag!=b'LDQ1':raise SystemExit(2)
 x,y,z=struct.unpack_from('<fff',data)
 if x < -900:raise SystemExit(3)
 time.sleep(.04)
 box=struct.pack('<9f',x,y,z,4,2,1.5,.25,.8,0)
 send(reply.pack(b'LDR1',revision,1,40,2)+b'ok'+box)
)PY";
    std::ofstream(folder/"detector.cfg")<<"python="<<argv[1]<<"\nmodel_dir="<<temporary.u8string()<<"\nprovider=cpu\n";
    {
      LearnedDetector detector;detector.Start(temporary);
      Wait([&]{auto status=detector.Status();Check(!status.failed,status.message.c_str());return status.ready;});
      detector.Submit(42,{{3,4,1,0}},.35f,1.5707963268f);
      std::optional<DetectorResult> result;
      Wait([&]{result=detector.Poll();return result.has_value();});
      Check(result->revision==42 && result->detections.size()==1 && result->detail=="ok","Fragmented protocol reply lost data");
      const auto & box=result->detections[0].oriented_bounds;
      Check(std::abs(box.center.x-3)<1e-4 && std::abs(box.center.y-4)<1e-4,"Display yaw / model heading round trip is wrong");
      Check(box.valid && std::abs(box.axis_x.x-std::cos(.25f+1.5707963268f))<1e-4,"Learned yaw did not reach the oriented box");
      for (std::uint64_t revision=43;revision<=100;++revision) detector.Submit(revision,{{1,2,3,0}},.35f,0);
      Wait([&]{if (auto next=detector.Poll()) return next->revision==100;return false;});
      detector.Submit(101,{{-999,0,0,0}},.35f,0);
      Wait([&]{return detector.Status().failed;});
      detector.Stop();
      // Stopping during a blocked model startup must not hang application exit.
      std::ofstream(worker)<<"import time\ntime.sleep(60)\n";
      detector.Start(temporary);std::this_thread::sleep_for(std::chrono::milliseconds(60));
      const auto start=std::chrono::steady_clock::now();detector.Stop();
      Check(std::chrono::steady_clock::now()-start<std::chrono::seconds(2),"Canceling model startup blocked UI shutdown");
    }
    const float invalid[]={0,0,0,-1,1,1,0,.9f,0};bool rejected=false;
    try {algorithms::DecodeCenterPointBox(invalid,0);} catch (...) {rejected=true;}
    Check(rejected,"Invalid neural box dimensions accepted");
    algorithms::DetectionParams filters;
    algorithms::Detection car,person,bicycle;car.class_id=0;person.class_id=3;bicycle.class_id=4;
    car.confidence=person.confidence=bicycle.confidence=.8f;
    Check(filters.Accepts(car) && filters.Accepts(person) && filters.Accepts(bicycle),"Default targets omit a supported class");
    filters.classes=1u<<3;
    Check(filters.Accepts(person) && !filters.Accepts(car) && !filters.Accepts(bicycle),"Person selection leaks other classes");
    filters.min_confidence=.9f;Check(!filters.Accepts(person),"Class selection bypasses the score filter");
    filters.classes=0;filters.min_confidence=.1f;Check(!filters.Accepts(person),"No targets still displays a class");
    std::filesystem::remove(folder/"detector.cfg");
    {LearnedDetector missing;missing.Start(temporary);Check(!missing.Status().configured,"Missing runtime was presented as working detection");}
    ConfigEnvironment(previous);std::filesystem::remove_all(temporary.parent_path());
    std::cout<<"learned_detector_smoke: protocol, latest request, yaw, process failure and cancellation passed\n";return 0;
  } catch (const std::exception & error) {
    ConfigEnvironment(previous);if (!temporary.empty()) std::filesystem::remove_all(temporary.parent_path());
    std::cerr<<error.what()<<'\n';return 1;
  }
}

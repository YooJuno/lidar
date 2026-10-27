#include "lidar_viewer/settings_io.hpp"
#include "lidar_viewer/platform.hpp"

#include <csignal>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#ifndef _WIN32
#include <sys/resource.h>
#endif

namespace
{
namespace fs = std::filesystem;
using namespace lidar_viewer::settings;

void Check(bool value, const char * message)
{
  if (!value) {
    throw std::runtime_error(message);
  }
}

template<typename Function>
void Throws(Function function, const char * message)
{
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

std::string Read(const fs::path & path)
{
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void TestNumbers()
{
  Check(ParseFiniteFloat("  -.57e1 \t") == -5.7f, "Valid finite scientific float rejected");
  Check(ParseStrictInt(" +42 \t") == 42, "Valid integer rejected");
  for (const auto * value : {"nan", "inf", "-Infinity", "1--", "1e", "0.5m", "", "1e999"}) {
    Throws([&] {ParseFiniteFloat(value);}, "Malformed or nonfinite float accepted");
  }
  for (const auto * value : {"1.2", "1x", "1 2", "", "9999999999999999999999"}) {
    Throws([&] {ParseStrictInt(value);}, "Malformed integer accepted");
  }
}

void TestFiles(const fs::path & folder)
{
  const auto target = folder / "profile.cfg";
  AtomicWriteText(target, "old\n");
  Check(Read(target) == "old\n", "Initial settings file not written");
  AtomicWriteText(target, "new\n");
  Check(Read(target) == "new\n", "Existing settings file not replaced");

#ifndef _WIN32
  const auto link = folder / "linked.cfg";
  fs::create_symlink(target, link);
  Throws([&] {AtomicWriteText(link, "bad");}, "Settings symlink was replaced");
  Check(fs::is_symlink(link) && Read(target) == "new\n", "Rejected symlink write changed original");
#endif
  Throws([&] {AtomicWriteText(folder, "bad");}, "Settings directory was replaced");

  // Force a partial write, then an EFBIG error in this test process only.
  // The previous complete profile must survive and the owned temp must vanish.
#ifndef _WIN32
  struct rlimit previous{};
  Check(::getrlimit(RLIMIT_FSIZE, &previous) == 0, "Cannot inspect test file-size limit");
  auto limited = previous;
  limited.rlim_cur = 3;
  const auto previous_signal = std::signal(SIGXFSZ, SIG_IGN);
  Check(::setrlimit(RLIMIT_FSIZE, &limited) == 0, "Cannot set test file-size limit");
  bool failed = false;
  try {
    AtomicWriteText(target, "This write must not replace the previous profile");
  } catch (const std::exception &) {
    failed = true;
  }
  const int restored = ::setrlimit(RLIMIT_FSIZE, &previous);
  std::signal(SIGXFSZ, previous_signal);
  Check(restored == 0 && failed, "Partial-write failure was not detected");
  Check(Read(target) == "new\n", "Failed partial write corrupted previous settings");
#else
  const HANDLE locked=CreateFileW(target.c_str(),GENERIC_READ,FILE_SHARE_READ,nullptr,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,nullptr);
  Check(locked!=INVALID_HANDLE_VALUE,"Cannot lock settings fixture");
  bool failed=false;
  try {AtomicWriteText(target,"must not replace locked settings");} catch (const std::exception &) {failed=true;}
  CloseHandle(locked);
  Check(failed && Read(target)=="new\n","Failed replacement corrupted locked settings");
#endif
  for (const auto & entry : fs::directory_iterator(folder)) {
    Check(entry.path().filename().string().find(".tmp-") == std::string::npos,
      "Temporary settings file leaked after failure");
  }
}
}  // namespace

int main()
{
  const fs::path folder=lidar_viewer::TemporaryDirectory("lidar-settings-test-");
  try {
    TestNumbers();
    TestFiles(folder);
    fs::remove_all(folder);
    std::cout << "settings_io_smoke: all checks passed\n";
    return 0;
  } catch (const std::exception & error) {
    fs::remove_all(folder);
    std::cerr << "settings_io_smoke: " << error.what() << '\n';
    return 1;
  }
}

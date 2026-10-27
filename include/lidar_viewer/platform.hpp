#pragma once
#include <chrono>
#include <cstdint>
#include <ctime>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#include <unistd.h>
#else
#include <unistd.h>
#endif

namespace lidar_viewer
{
inline std::filesystem::path UserHome()
{
#ifdef _WIN32
  if (const char * home=std::getenv("USERPROFILE")) return std::filesystem::u8path(home);
#endif
  if (const char * home=std::getenv("HOME")) return std::filesystem::u8path(home);
  return std::filesystem::current_path();
}
inline std::filesystem::path UserConfigurationDirectory()
{
  if (const char * config=std::getenv("XDG_CONFIG_HOME"); config && *config && std::filesystem::u8path(config).is_absolute())
    return std::filesystem::u8path(config)/"lidar_viewer";
#ifdef _WIN32
  if (const char * data=std::getenv("APPDATA")) return std::filesystem::u8path(data)/"LiDARViewer";
  return UserHome()/"AppData/Roaming/LiDARViewer";
#elif defined(__APPLE__)
  return UserHome()/"Library/Application Support/LiDARViewer";
#else
  return UserHome()/".config/lidar_viewer";
#endif
}
inline std::uint64_t ProcessId()
{
#ifdef _WIN32
  return GetCurrentProcessId();
#else
  return static_cast<std::uint64_t>(getpid());
#endif
}
inline std::filesystem::path ExecutablePath()
{
#ifdef _WIN32
  std::vector<wchar_t> path(32768);
  const auto length=GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  return length && length<path.size() ? std::filesystem::path(std::wstring(path.data(),length)) : std::filesystem::path{};
#elif defined(__APPLE__)
  std::uint32_t length=0; _NSGetExecutablePath(nullptr,&length);
  std::vector<char> path(length);
  return _NSGetExecutablePath(path.data(),&length)==0 ? std::filesystem::weakly_canonical(path.data()) : std::filesystem::path{};
#else
  std::error_code error;
  return std::filesystem::read_symlink("/proc/self/exe",error);
#endif
}
inline std::tm UtcTime(std::time_t value)
{
  std::tm result{};
#ifdef _WIN32
  gmtime_s(&result,&value);
#else
  gmtime_r(&value,&result);
#endif
  return result;
}
inline std::filesystem::path TemporaryDirectory(const std::string & prefix)
{
  const auto stamp=std::chrono::steady_clock::now().time_since_epoch().count();
  for (int attempt=0;attempt<100;++attempt) {
    const auto path=std::filesystem::temp_directory_path()/(prefix+std::to_string(ProcessId())+"-"+
      std::to_string(stamp)+"-"+std::to_string(attempt));
    if (std::filesystem::create_directory(path)) return path;
  }
  throw std::runtime_error("Cannot create private temporary directory");
}
}  // namespace lidar_viewer

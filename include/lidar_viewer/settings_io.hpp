#pragma once

#include <cerrno>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifdef _WIN32
#include "lidar_viewer/platform.hpp"
#include <atomic>
#else
#include <unistd.h>
#endif

namespace lidar_viewer::settings
{
inline void RequireWholeToken(const std::string & text, std::size_t consumed)
{
  for (; consumed < text.size(); ++consumed) {
    if (!std::isspace(static_cast<unsigned char>(text[consumed]))) {
      throw std::invalid_argument("Unexpected characters after numeric value");
    }
  }
}

inline float ParseFiniteFloat(const std::string & text)
{
  std::size_t consumed = 0;
  const float value = std::stof(text, &consumed);
  RequireWholeToken(text, consumed);
  if (!std::isfinite(value)) {
    throw std::invalid_argument("Numeric value must be finite");
  }
  return value;
}

inline int ParseStrictInt(const std::string & text)
{
  std::size_t consumed = 0;
  const int value = std::stoi(text, &consumed);
  RequireWholeToken(text, consumed);
  return value;
}

// Flush and close a privately-created same-directory file before the atomic
// rename. Nothing after a successful rename can turn success into an error.
// This is an atomic replacement, not a directory-fsync power-loss guarantee.
inline void AtomicWriteText(const std::filesystem::path & destination, std::string_view contents)
{
  namespace fs = std::filesystem;
  if (destination.filename().empty() || destination.filename() == "." ||
      destination.filename() == "..") {
    throw std::invalid_argument("Settings destination must name a file");
  }
  const auto parent = destination.has_parent_path() ? destination.parent_path() : fs::path(".");
  std::error_code error;
  const auto existing = fs::symlink_status(destination, error);
  if (error && error != std::errc::no_such_file_or_directory) {
    throw std::system_error(error, "Cannot inspect settings destination");
  }
  if (fs::is_symlink(existing) || (fs::exists(existing) && !fs::is_regular_file(existing))) {
    throw std::runtime_error("Refusing to replace a settings symlink or non-regular file");
  }
  if (!fs::is_directory(parent)) {
    throw std::runtime_error("Settings parent directory does not exist");
  }
#ifdef _WIN32
  static std::atomic<std::uint64_t> serial{0};
  const auto temporary=parent/(L"."+destination.filename().wstring()+L".tmp-"+
    std::to_wstring(ProcessId())+L"-"+std::to_wstring(std::chrono::steady_clock::now().time_since_epoch().count())+
    L"-"+std::to_wstring(serial.fetch_add(1)));
  HANDLE file=CreateFileW(temporary.c_str(),GENERIC_WRITE,0,nullptr,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,nullptr);
  if (file==INVALID_HANDLE_VALUE) throw std::system_error(GetLastError(),std::system_category(),"Cannot create temporary settings file");
  try {
    std::size_t offset=0;
    while (offset<contents.size()) {
      DWORD written=0;
      const auto length=static_cast<DWORD>(std::min<std::size_t>(contents.size()-offset, 0x7fffffffu));
      if (!WriteFile(file,contents.data()+offset,length,&written,nullptr) || !written)
        throw std::system_error(GetLastError(),std::system_category(),"Cannot write settings");
      offset+=written;
    }
    if (!FlushFileBuffers(file)) throw std::system_error(GetLastError(),std::system_category(),"Cannot flush settings");
    const HANDLE closing=file; file=INVALID_HANDLE_VALUE;
    if (!CloseHandle(closing)) throw std::system_error(GetLastError(),std::system_category(),"Cannot close settings");
    if (!MoveFileExW(temporary.c_str(),destination.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))
      throw std::system_error(GetLastError(),std::system_category(),"Cannot replace settings");
  } catch (...) {
    if (file!=INVALID_HANDLE_VALUE) CloseHandle(file);
    std::error_code ignored; fs::remove(temporary,ignored); throw;
  }
#else
  const std::string pattern = (parent / ("." + destination.filename().string() + ".tmp-XXXXXX")).string();
  std::vector<char> name(pattern.begin(), pattern.end());
  name.push_back('\0');
  int descriptor = ::mkstemp(name.data());
  if (descriptor < 0) {
    throw std::system_error(errno, std::generic_category(), "Cannot create temporary settings file");
  }
  const fs::path temporary(name.data());
  try {
    std::size_t offset = 0;
    while (offset < contents.size()) {
      const auto count = ::write(descriptor, contents.data() + offset, contents.size() - offset);
      if (count < 0 && errno == EINTR) {
        continue;
      }
      if (count <= 0) {
        throw std::system_error(count < 0 ? errno : EIO, std::generic_category(), "Cannot write settings");
      }
      offset += static_cast<std::size_t>(count);
    }
    if (::fsync(descriptor) != 0) {
      throw std::system_error(errno, std::generic_category(), "Cannot flush settings");
    }
    const int closing = descriptor;
    descriptor = -1;
    if (::close(closing) != 0) {
      throw std::system_error(errno, std::generic_category(), "Cannot close settings");
    }
    fs::rename(temporary, destination);
  } catch (...) {
    if (descriptor >= 0) {
      ::close(descriptor);
    }
    std::error_code ignored;
    fs::remove(temporary, ignored);
    throw;
  }
#endif
}
}  // namespace lidar_viewer::settings

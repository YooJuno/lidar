#include "lidar_viewer/bag_catalog.hpp"
#include "lidar_viewer/settings_io.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <regex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace
{
namespace fs = std::filesystem;

bool IsStorage(const fs::path & path)
{
  return path.extension() == ".mcap" || path.extension() == ".db3";
}

bool IsBagDirectory(const fs::path & path)
{
  std::error_code error;
  return fs::is_directory(path, error) &&
         fs::is_regular_file(path / "metadata.yaml", error);
}

fs::path Canonical(const fs::path & path)
{
  std::error_code error;
  auto result = fs::canonical(path, error);
  return error ? fs::path{} : result;
}

fs::path AbsoluteIdentity(const fs::path & path)
{
  const auto text = path.u8string();
  if (text.empty() || text.find_first_of("\r\n") != std::string::npos ||
      text.find('\0') != std::string::npos) {
    throw std::invalid_argument("Data folder paths must not be empty or contain line breaks/NUL");
  }
  // Resolve existing aliases and parent components, retaining unavailable
  // folders as absolute paths for removable or temporarily offline storage.
  return fs::weakly_canonical(fs::absolute(path)).lexically_normal();
}

void SortSources(std::vector<BagSource> & sources)
{
  std::sort(sources.begin(), sources.end(), [](const BagSource & left, const BagSource & right) {
    if (left.recording_date != right.recording_date) return left.recording_date > right.recording_date;
    if (left.recording_time != right.recording_time) return left.recording_time > right.recording_time;
    return left.path < right.path;
  });
}

std::vector<fs::path> NormalizeFolders(const std::vector<fs::path> & folders)
{
  std::vector<fs::path> result;
  std::set<fs::path> seen;
  for (const auto & folder : folders) {
    const auto path = AbsoluteIdentity(folder);
    std::error_code error;
    const bool exists = fs::exists(path, error);
    // Inaccessible/offline storage stays in the library. Reject a reachable
    // regular file rather than silently treating it as a directory.
    if (!error && exists && !fs::is_directory(path)) {
      throw std::invalid_argument("Data folder is not a directory: " + path.string());
    }
    if (seen.insert(path).second) result.push_back(path);
  }
  return result;
}

fs::path ContainingBag(fs::path path)
{
  while (!path.empty()) {
    if (IsBagDirectory(path)) return Canonical(path);
    const auto parent = path.parent_path();
    if (parent == path) break;
    path = parent;
  }
  return {};
}

void ReadFilenameDate(BagSource & source)
{
  // Recording names used by sensor_collection. Do not mistake an unrelated
  // filesystem timestamp (e.g. copying a bag today) for its acquisition date.
  static const std::regex pattern(
    R"(^(?:sensors|sensor_smoke)_([0-9]{8})_([0-9]{6})(?:_.+)?$)");
  const auto name = source.path.has_extension() ? source.path.stem().string() :
    source.path.filename().string();
  std::smatch match;
  if (!std::regex_match(name, match, pattern)) return;
  const std::string date = match[1], time = match[2];
  const int year = std::stoi(date.substr(0, 4));
  const int month = std::stoi(date.substr(4, 2));
  const int day = std::stoi(date.substr(6, 2));
  const int hour = std::stoi(time.substr(0, 2));
  const int minute = std::stoi(time.substr(2, 2));
  const int second = std::stoi(time.substr(4, 2));
  if (year < 1 || month < 1 || month > 12 || hour > 23 || minute > 59 || second > 59) return;
  static constexpr int month_days[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
  int last_day = month_days[month - 1];
  if (month == 2 && year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)) ++last_day;
  if (day < 1 || day > last_day) return;
  source.recording_date = date.substr(0, 4) + "-" + date.substr(4, 2) + "-" + date.substr(6, 2);
  source.recording_time = time.substr(0, 2) + ":" + time.substr(2, 2) + ":" + time.substr(4, 2);
}
}  // namespace

std::vector<BagSource> DiscoverBagSources(const std::filesystem::path & input)
{
  namespace fs = std::filesystem;
  std::vector<BagSource> sources;
  const auto selected = Canonical(input);
  if (selected.empty()) return sources;
  std::error_code error;
  const bool is_directory = fs::is_directory(selected, error);
  if (!is_directory && selected.filename() != "metadata.yaml" && !IsStorage(selected)) return sources;

  const auto selected_bag = ContainingBag(is_directory ? selected : selected.parent_path());
  if (!selected_bag.empty() || !is_directory) {
    BagSource source;
    source.path = selected_bag.empty() ? selected : selected_bag;
    source.display_name = source.path.filename().u8string();
    ReadFilenameDate(source);
    return {std::move(source)};
  }
  const auto root = selected;
  std::set<fs::path> seen_sources;
  std::set<fs::path> seen_directories;
  std::vector<fs::path> pending{root};
  auto add = [&](const fs::path & path) {
    auto canonical = Canonical(path);
    // A loose symlink may point at a chunk inside a recorded bag. Normalize
    // it too, so aliases cannot expose individual chunks as extra recordings.
    if (!canonical.empty() && IsStorage(canonical)) {
      const auto bag = ContainingBag(canonical.parent_path());
      if (!bag.empty()) canonical = bag;
    }
    if (canonical.empty() || !seen_sources.insert(canonical).second) return;
    BagSource source;
    source.path = canonical;
    source.display_name = canonical.filename().u8string();
    ReadFilenameDate(source);
    sources.push_back(std::move(source));
  };
  while (!pending.empty()) {
    const auto directory = pending.back();
    pending.pop_back();
    const auto canonical = Canonical(directory);
    if (canonical.empty() || !seen_directories.insert(canonical).second) continue;
    fs::directory_iterator entry(directory, fs::directory_options::skip_permission_denied, error), end;
    while (entry != end) {
      const auto path = entry->path();
      error.clear();
      if (entry->is_directory(error)) {
        if (IsBagDirectory(path)) {
          add(path);
        } else if (!entry->is_symlink(error)) {
          pending.push_back(path);
        }
      } else if (entry->is_regular_file(error) && IsStorage(path)) {
        add(path);
      }
      error.clear();
      entry.increment(error);
      if (error) break;
    }
  }
  SortSources(sources);
  return sources;
}

std::vector<BagSource> MergeBagSources(
  std::vector<BagSource> left, const std::vector<BagSource> & right)
{
  left.insert(left.end(), right.begin(), right.end());
  std::vector<BagSource> merged;
  std::set<std::filesystem::path> seen;
  for (auto & source : left) {
    source.path = AbsoluteIdentity(source.path);
    if (seen.insert(source.path).second) merged.push_back(std::move(source));
  }
  SortSources(merged);
  return merged;
}

std::vector<std::filesystem::path> LoadBagFolders(const std::filesystem::path & config)
{
  namespace fs = std::filesystem;
  std::error_code error;
  if (!fs::exists(config, error)) {
    if (error) throw std::system_error(error, "Cannot inspect data folder library");
    return {};
  }
  std::ifstream input(config);
  if (!input) throw std::runtime_error("Cannot read data folder library: " + config.string());
  std::vector<fs::path> folders;
  std::string line;
  std::size_t number = 0;
  while (std::getline(input, line)) {
    ++number;
    std::istringstream fields(line);
    fields >> std::ws;
    if (fields.eof() || fields.peek() == '#') continue;
    std::string key, value;
    fields >> key >> std::ws;
    if (key != "folder" || fields.peek() != '"' || !(fields >> std::quoted(value))) {
      throw std::runtime_error("Malformed data folder library at line " + std::to_string(number));
    }
    fields >> std::ws;
    if (!fields.eof() || !fs::u8path(value).is_absolute()) {
      throw std::runtime_error("Invalid data folder path at line " + std::to_string(number));
    }
    folders.push_back(fs::u8path(value));
  }
  if (input.bad()) throw std::runtime_error("Failed to read data folder library: " + config.string());
  return NormalizeFolders(folders);
}

void SaveBagFolders(
  const std::filesystem::path & config,
  const std::vector<std::filesystem::path> & folders)
{
  namespace fs = std::filesystem;
  const auto normalized = NormalizeFolders(folders);
  std::ostringstream contents;
  contents << "# LiDAR Studio data folders v1\n";
  for (const auto & folder : normalized) contents << "folder " << std::quoted(folder.u8string()) << '\n';
  if (config.has_parent_path()) fs::create_directories(config.parent_path());
  lidar_viewer::settings::AtomicWriteText(config, contents.str());
}

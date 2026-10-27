#include "lidar_viewer/bag_catalog.hpp"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#ifndef _WIN32
#include <sys/resource.h>
#endif

namespace
{
namespace fs = std::filesystem;

void Check(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

template<typename Function>
void Throws(Function function, const char * message)
{
  try {function();} catch (const std::exception &) {return;}
  throw std::runtime_error(message);
}

std::string Read(const fs::path & path)
{
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

void File(const fs::path & path)
{
  fs::create_directories(path.parent_path());
  std::ofstream output(path);
  output << "fixture\n";
}

fs::path Bag(const fs::path & root, const char * name)
{
  const auto path = root / name;
  File(path / "metadata.yaml");
  File(path / "chunk_0.mcap");
  File(path / "chunk_1.mcap");
  return path;
}

void TestLibrary(const fs::path & root)
{
  const auto config = root / "config" / "data_folders.cfg";
#ifdef _WIN32
  const auto folder = root / "folder with spaces";
#else
  const auto folder = root / "folder with spaces and \"quotes\" \\ slash";
#endif
  fs::create_directory(folder);
  const auto offline = root / "disconnected" / "bag folder";
  Check(LoadBagFolders(config).empty(), "Missing library should return an empty list");
  SaveBagFolders(config, {folder, offline, folder / "."});
  const auto loaded = LoadBagFolders(config);
  Check(loaded.size() == 2 && loaded[0] == fs::canonical(folder) && loaded[1] == fs::weakly_canonical(offline),
    "Library lost quoted/offline paths or did not deduplicate");
  const auto previous = Read(config);
  Throws([&] {SaveBagFolders(config, {folder, fs::path("bad\nfolder")});},
    "Newline in a library path was accepted");
  Throws([&] {SaveBagFolders(config, {root / "frame.bin"});},
    "A regular file was accepted as a library folder");
  Check(Read(config) == previous, "Invalid save clobbered the previous library");
  for (const auto * invalid : {"folder unquoted\n", "folder \"relative\"\n",
      "folder \"/tmp/path\" trailing\n", "unknown \"/tmp/path\"\n", "folder \"/tmp/unclosed\n"}) {
    std::ofstream(config) << invalid;
    Throws([&] {LoadBagFolders(config);}, "Malformed library entry was accepted");
  }
  SaveBagFolders(config, {folder});
#ifndef _WIN32
  const auto before_failure = Read(config);
  struct rlimit previous_limit{};
  Check(::getrlimit(RLIMIT_FSIZE, &previous_limit) == 0, "Cannot inspect file size limit");
  auto limited = previous_limit;
  limited.rlim_cur = 3;
  const auto previous_signal = std::signal(SIGXFSZ, SIG_IGN);
  Check(::setrlimit(RLIMIT_FSIZE, &limited) == 0, "Cannot set file size limit");
  bool failed = false;
  try {SaveBagFolders(config, {folder, offline});} catch (const std::exception &) {failed = true;}
  const int restored = ::setrlimit(RLIMIT_FSIZE, &previous_limit);
  std::signal(SIGXFSZ, previous_signal);
  Check(restored == 0 && failed, "Atomic write failure was not reported");
  Check(Read(config) == before_failure, "Partial library write corrupted the prior complete file");
#endif
  for (const auto & entry : fs::directory_iterator(config.parent_path())) {
    Check(entry.path().filename().string().find(".tmp-") == std::string::npos,
      "Failed library save leaked a temporary file");
  }
}
}  // namespace

int main()
{
  const auto root = fs::temp_directory_path() / ("lidar-bag-catalog-" +
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    fs::create_directories(root);
    const auto latest = Bag(root, "sensors_20261023_120204_157971");
    const auto earlier = Bag(root, "sensors_20261023_102818_576475");
    const auto previous = Bag(root, "sensor_smoke_20261022_190600");
    const auto unknown = Bag(root, "unknown_recording");
    const auto nested = Bag(root / "archive", "sensors_20261021_103005_1");
    File(root / "loose" / "sensors_20261020_160000_0.mcap");
    File(root / "loose" / "sensors_20261020_150000_0.db3");
    File(root / "frame.bin");
    // A bag-looking subdirectory inside a bag is part of the same indivisible
    // recording, not another source.
    Bag(latest / "internal", "sensors_20261026_120000_1");
    std::error_code error;
    fs::create_directory_symlink(latest, root / "latest_alias", error);
    error.clear();
    fs::create_directory_symlink(root, root / "archive" / "loop", error);
    error.clear();
    fs::create_symlink(latest / "chunk_0.mcap", root / "chunk_alias.mcap", error);

    const auto all = DiscoverBagSources(root);
    Check(all.size() == 7, "Collection discovery duplicates chunks/aliases or loses nested bags/storage");
    Check(all.front().path == fs::canonical(latest), "Newest recording was not sorted first");
    Check(all[1].path == fs::canonical(earlier), "Same-day recordings were not sorted by time");
    Check(all[2].path == fs::canonical(previous), "Previous-day recording was not recognized");
    Check(all.back().path == fs::canonical(unknown) && all.back().recording_date.empty(),
      "Unknown dates should be last and should not use file modification time");
    Check(all.front().recording_date == "2026-10-23" && all.front().recording_time == "12:02:04",
      "Filename date/time formatting is incorrect");
    Check(all.front().display_name == latest.filename(), "Display name changed the original source name");
    const auto selected = DiscoverBagSources(latest);
    Check(selected.size() == 1 && selected.front().path == fs::canonical(latest),
      "A selected bag should list only itself");
    const auto metadata = DiscoverBagSources(latest / "metadata.yaml");
    const auto split = DiscoverBagSources(latest / "chunk_1.mcap");
    Check(metadata.size() == selected.size() && split.size() == selected.size(),
      "Metadata or direct split storage failed to normalize to the bag root");
    Check(metadata.front().path == selected.front().path && split.front().path == selected.front().path,
      "Split storage was listed as a separate recording");
    const auto loose = DiscoverBagSources(root / "loose" / "sensors_20261020_160000_0.mcap");
    Check(loose.size() == 1 && loose.front().path.extension() == ".mcap",
      "A selected loose storage file should list only itself");
    Check(DiscoverBagSources(root / "frame.bin").empty(), "BIN input should not request a bag catalog");
    Check(DiscoverBagSources(root / "missing").empty(), "Missing input should return an empty catalog");
    fs::create_directory(root / "empty");
    Check(DiscoverBagSources(root / "empty").empty(), "Empty collection should return an empty catalog");

    const auto dates = root / "dates";
    Bag(dates, "sensors_20240229_235959_valid");
    Bag(dates, "sensors_20230229_235959_invalid");
    Bag(dates, "sensors_20261032_000000_invalid");
    Bag(dates, "sensors_20261023_240000_invalid");
    Bag(dates, "sensors_20261301_120000_invalid");
    Bag(dates, "sensors_20261023_126000_invalid");
    Bag(dates, "sensors_20261023_120060_invalid");
    Bag(dates, "sensors_20261023_120000junk");
    const auto validated = DiscoverBagSources(dates);
    Check(validated.size() == 8 && validated.front().recording_date == "2024-02-29",
      "Valid leap-day recording was not recognized");
    Check(std::count_if(validated.begin(), validated.end(), [](const BagSource & source) {
      return !source.recording_date.empty();
    }) == 1, "Malformed recording dates/times were accepted");
    auto overlap = all;
    overlap.front().path /= ".";
    const auto merged = MergeBagSources(overlap, all);
    Check(merged.size() == all.size() && merged.front().path == fs::canonical(latest) &&
      merged.back().recording_date.empty(), "Merged catalogs were duplicated or sorted incorrectly");
    TestLibrary(root);
    fs::remove_all(root);
    std::cout << "Bag catalog smoke test passed\n";
    return 0;
  } catch (const std::exception & error) {
    fs::remove_all(root);
    std::cerr << error.what() << '\n';
    return 1;
  }
}

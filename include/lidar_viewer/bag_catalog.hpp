#pragma once

#include <filesystem>
#include <string>
#include <vector>

struct BagSource
{
  std::filesystem::path path;
  std::string display_name;
  // These are derived from a recognized recording filename, not the bag clock
  // or a file's modification time. Empty means that the date is unknown.
  std::string recording_date;
  std::string recording_time;
};

// A selected bag (including metadata or a split storage file) returns only that
// bag. A collection directory discovers nested bags and loose storage files.
// Bag directories are indivisible: their individual chunks are never listed.
std::vector<BagSource> DiscoverBagSources(const std::filesystem::path & input);

// De-duplicate by normalized absolute path and apply the recording date sort.
std::vector<BagSource> MergeBagSources(
  std::vector<BagSource> left, const std::vector<BagSource> & right);

// The persistent library uses quoted absolute paths. Temporarily unavailable
// directories remain listed, so an unplugged drive does not erase a source.
// A missing config returns an empty list; malformed config throws.
std::vector<std::filesystem::path> LoadBagFolders(const std::filesystem::path & config);
void SaveBagFolders(
  const std::filesystem::path & config,
  const std::vector<std::filesystem::path> & folders);

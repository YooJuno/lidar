#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr float kPi = 3.14159265358979323846f;
constexpr int kOusterLegacyChannels = 64;
constexpr int kOusterLegacyBlocksPerPacket = 16;
constexpr int kOusterLegacyColumnsPerFrame = 1024;
constexpr int kOusterLegacyPacketBytes = 12608;
constexpr int kOusterLegacyBlockBytes = 788;
constexpr int kOusterLegacyChannelBytes = 12;
constexpr int kOusterLegacyRecordTypeLidar = 2;
constexpr std::uint32_t kOusterLegacyRecordBytes = 806912;
constexpr std::uint32_t kOusterLegacyValidStatus = 0xFFFFFFFFu;

struct Vec3
{
  float x{0.0f};
  float y{0.0f};
  float z{0.0f};

  Vec3 operator+(const Vec3 & other) const { return {x + other.x, y + other.y, z + other.z}; }
  Vec3 operator-(const Vec3 & other) const { return {x - other.x, y - other.y, z - other.z}; }
  Vec3 operator*(float scalar) const { return {x * scalar, y * scalar, z * scalar}; }
};

struct KittiPoint
{
  float x;
  float y;
  float z;
  float intensity;
};

enum class InputFormat
{
  KittiXyzi,
  OusterLegacyContainer,
};

struct FrameHandle
{
  fs::path path;
  InputFormat format{InputFormat::KittiXyzi};
  std::uint64_t payload_offset{0};
  std::uint32_t payload_size{0};
  std::string label;
};

struct FrameData
{
  std::vector<KittiPoint> points;
  Vec3 center{};
  float radius{30.0f};
  float min_z{0.0f};
  float max_z{0.0f};
  float ground_z{0.0f};
};

std::vector<FrameHandle> ResolveInputFrames(const fs::path & input);
FrameData LoadFrame(const FrameHandle & handle);

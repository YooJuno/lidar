#include "lidar_viewer/data_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace
{
constexpr float kOusterBeamOriginOffsetM = 0.015806f;
constexpr float kOusterEncoderTicksPerRev = 90112.0f;

constexpr std::array<float, kOusterLegacyChannels> kOs1_64AltitudeDeg{{
  16.611f,  16.084f,  15.557f,  15.029f,  14.502f,  13.975f,  13.447f,  12.920f,
  12.393f,  11.865f,  11.338f,  10.811f,  10.283f,  9.756f,   9.229f,   8.701f,
  8.174f,   7.646f,   7.119f,   6.592f,   6.064f,   5.537f,   5.010f,   4.482f,
  3.955f,   3.428f,   2.900f,   2.373f,   1.846f,   1.318f,   0.791f,   0.264f,
  -0.264f,  -0.791f,  -1.318f,  -1.846f,  -2.373f,  -2.900f,  -3.428f,  -3.955f,
  -4.482f,  -5.010f,  -5.537f,  -6.064f,  -6.592f,  -7.119f,  -7.646f,  -8.174f,
  -8.701f,  -9.229f,  -9.756f,  -10.283f, -10.811f, -11.338f, -11.865f, -12.393f,
  -12.920f, -13.447f, -13.975f, -14.502f, -15.029f, -15.557f, -16.084f, -16.611f
}};

constexpr std::array<float, kOusterLegacyChannels> kOs1_64AzimuthOffsetDeg{{
  3.164f, 1.055f, -1.055f, -3.164f, 3.164f, 1.055f, -1.055f, -3.164f,
  3.164f, 1.055f, -1.055f, -3.164f, 3.164f, 1.055f, -1.055f, -3.164f,
  3.164f, 1.055f, -1.055f, -3.164f, 3.164f, 1.055f, -1.055f, -3.164f,
  3.164f, 1.055f, -1.055f, -3.164f, 3.164f, 1.055f, -1.055f, -3.164f,
  3.164f, 1.055f, -1.055f, -3.164f, 3.164f, 1.055f, -1.055f, -3.164f,
  3.164f, 1.055f, -1.055f, -3.164f, 3.164f, 1.055f, -1.055f, -3.164f,
  3.164f, 1.055f, -1.055f, -3.164f, 3.164f, 1.055f, -1.055f, -3.164f,
  3.164f, 1.055f, -1.055f, -3.164f, 3.164f, 1.055f, -1.055f, -3.164f
}};

void FinalizeFrame(FrameData & frame)
{
  if (frame.points.empty()) {
    frame.center = {0.0f, 0.0f, 0.0f};
    frame.radius = 10.0f;
    frame.min_z = 0.0f;
    frame.max_z = 0.0f;
    frame.ground_z = 0.0f;
    return;
  }

  Vec3 min_corner{
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max(),
    std::numeric_limits<float>::max()
  };
  Vec3 max_corner{
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest(),
    std::numeric_limits<float>::lowest()
  };
  float min_intensity = std::numeric_limits<float>::max();
  float max_intensity = std::numeric_limits<float>::lowest();
  std::vector<float> z_samples;
  z_samples.reserve(frame.points.size());

  for (auto & p : frame.points) {
    min_corner.x = std::min(min_corner.x, p.x);
    min_corner.y = std::min(min_corner.y, p.y);
    min_corner.z = std::min(min_corner.z, p.z);
    max_corner.x = std::max(max_corner.x, p.x);
    max_corner.y = std::max(max_corner.y, p.y);
    max_corner.z = std::max(max_corner.z, p.z);
    min_intensity = std::min(min_intensity, p.intensity);
    max_intensity = std::max(max_intensity, p.intensity);
    z_samples.push_back(p.z);
  }

  frame.center = {
    0.5f * (min_corner.x + max_corner.x),
    0.5f * (min_corner.y + max_corner.y),
    0.5f * (min_corner.z + max_corner.z)
  };
  const Vec3 extent = max_corner - min_corner;
  frame.radius = std::max({extent.x, extent.y, extent.z}) * 0.7f;
  frame.radius = std::max(frame.radius, 10.0f);
  frame.min_z = min_corner.z;
  frame.max_z = max_corner.z;

  const std::size_t ground_index = std::min(
    z_samples.size() - 1,
    static_cast<std::size_t>(static_cast<double>(z_samples.size() - 1) * 0.12));
  std::nth_element(z_samples.begin(), z_samples.begin() + static_cast<std::ptrdiff_t>(ground_index), z_samples.end());
  frame.ground_z = z_samples[ground_index];

  const float intensity_range = max_intensity - min_intensity;
  if (intensity_range > 1e-6f) {
    for (auto & p : frame.points) {
      p.intensity = (p.intensity - min_intensity) / intensity_range;
    }
  } else {
    for (auto & p : frame.points) {
      p.intensity = 0.5f;
    }
  }
}

bool LooksLikeKittiXyzi(const fs::path & file_path)
{
  std::error_code ec;
  const auto size = fs::file_size(file_path, ec);
  return !ec && size > 0 && (size % sizeof(KittiPoint) == 0);
}

std::optional<std::vector<FrameHandle>> ScanOusterLegacyFrames(const fs::path & file_path)
{
  std::ifstream stream(file_path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Failed to open: " + file_path.string());
  }

  stream.seekg(0, std::ios::end);
  const std::uint64_t file_size = static_cast<std::uint64_t>(stream.tellg());
  stream.seekg(0, std::ios::beg);

  std::vector<FrameHandle> frames;
  std::uint64_t offset = 0;
  std::size_t lidar_index = 0;

  while ((offset + 8ull) <= file_size) {
    std::uint32_t type = 0;
    std::uint32_t size = 0;
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    stream.read(reinterpret_cast<char *>(&type), sizeof(type));
    stream.read(reinterpret_cast<char *>(&size), sizeof(size));
    if (!stream) {
      return std::nullopt;
    }

    const std::uint64_t next_offset = offset + 8ull + size;
    if (size == 0 || next_offset > file_size) {
      return std::nullopt;
    }

    if (type == kOusterLegacyRecordTypeLidar) {
      if (size != kOusterLegacyRecordBytes || (size % kOusterLegacyPacketBytes) != 0) {
        return std::nullopt;
      }
      std::ostringstream label;
      label << file_path.filename().string() << " #" << std::setw(4) << std::setfill('0') << lidar_index;
      frames.push_back(FrameHandle{
        file_path,
        InputFormat::OusterLegacyContainer,
        offset + 8ull,
        size,
        label.str()
      });
      ++lidar_index;
    }

    offset = next_offset;
  }

  if (offset != file_size || frames.empty()) {
    return std::nullopt;
  }
  return frames;
}

std::vector<FrameHandle> ResolveFramesFromFile(const fs::path & file_path)
{
  if (const auto ouster_frames = ScanOusterLegacyFrames(file_path)) {
    return *ouster_frames;
  }

  if (LooksLikeKittiXyzi(file_path)) {
    return {FrameHandle{file_path, InputFormat::KittiXyzi, 0, 0, file_path.filename().string()}};
  }

  throw std::runtime_error("Unsupported .bin format: " + file_path.string());
}

FrameData LoadKittiFrame(const fs::path & file_path)
{
  std::ifstream stream(file_path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Failed to open: " + file_path.string());
  }

  stream.seekg(0, std::ios::end);
  const std::streamsize size = stream.tellg();
  stream.seekg(0, std::ios::beg);

  if (size <= 0 || size % static_cast<std::streamsize>(sizeof(KittiPoint)) != 0) {
    throw std::runtime_error("Not a KITTI x,y,z,intensity file: " + file_path.string());
  }

  FrameData frame;
  const std::size_t count = static_cast<std::size_t>(size) / sizeof(KittiPoint);
  frame.points.resize(count);

  if (!stream.read(reinterpret_cast<char *>(frame.points.data()), size)) {
    throw std::runtime_error("Failed to read: " + file_path.string());
  }

  FinalizeFrame(frame);
  return frame;
}

FrameData LoadOusterLegacyFrame(const FrameHandle & handle)
{
  std::ifstream stream(handle.path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("Failed to open: " + handle.path.string());
  }

  stream.seekg(static_cast<std::streamoff>(handle.payload_offset), std::ios::beg);
  std::vector<std::uint8_t> payload(handle.payload_size);
  if (!stream.read(reinterpret_cast<char *>(payload.data()), static_cast<std::streamsize>(payload.size()))) {
    throw std::runtime_error("Failed to read Ouster frame: " + handle.path.string());
  }

  FrameData frame;
  frame.points.reserve(static_cast<std::size_t>(kOusterLegacyColumnsPerFrame) * kOusterLegacyChannels);

  for (std::size_t packet_index = 0; packet_index < (payload.size() / kOusterLegacyPacketBytes); ++packet_index) {
    const std::uint8_t * packet = payload.data() + packet_index * kOusterLegacyPacketBytes;
    for (int block = 0; block < kOusterLegacyBlocksPerPacket; ++block) {
      const std::uint8_t * block_ptr = packet + block * kOusterLegacyBlockBytes;

      std::uint32_t encoder_count = 0;
      std::uint32_t status = 0;
      std::memcpy(&encoder_count, block_ptr + 12, sizeof(encoder_count));
      std::memcpy(
        &status,
        block_ptr + 16 + kOusterLegacyChannels * kOusterLegacyChannelBytes,
        sizeof(status));
      if (status != kOusterLegacyValidStatus) {
        continue;
      }

      for (int row = 0; row < kOusterLegacyChannels; ++row) {
        const std::uint8_t * channel_ptr = block_ptr + 16 + row * kOusterLegacyChannelBytes;
        std::uint32_t word1 = 0;
        std::uint32_t word2 = 0;
        std::memcpy(&word1, channel_ptr, sizeof(word1));
        std::memcpy(&word2, channel_ptr + 4, sizeof(word2));

        const std::uint32_t range_mm = word1 & 0x000FFFFFu;
        if (range_mm == 0) {
          continue;
        }

        const float range_m = static_cast<float>(range_mm) * 0.001f;
        const float altitude = kOs1_64AltitudeDeg[static_cast<std::size_t>(row)] * kPi / 180.0f;
        const float azimuth_offset = -kOs1_64AzimuthOffsetDeg[static_cast<std::size_t>(row)] * kPi / 180.0f;
        const float encoder_theta = 2.0f * kPi * (1.0f - static_cast<float>(encoder_count) / kOusterEncoderTicksPerRev);
        const float beam_range = range_m - kOusterBeamOriginOffsetM;
        const float beam_theta = encoder_theta + azimuth_offset;

        frame.points.push_back(KittiPoint{
          beam_range * std::cos(beam_theta) * std::cos(altitude) +
            kOusterBeamOriginOffsetM * std::cos(encoder_theta),
          beam_range * std::sin(beam_theta) * std::cos(altitude) +
            kOusterBeamOriginOffsetM * std::sin(encoder_theta),
          beam_range * std::sin(altitude),
          static_cast<float>(word2 & 0xFFFFu)
        });
      }
    }
  }

  FinalizeFrame(frame);
  return frame;
}
}  // namespace

std::vector<FrameHandle> ResolveInputFrames(const fs::path & input)
{
  if (fs::is_regular_file(input)) {
    if (input.extension() != ".bin") {
      throw std::runtime_error("Expected .bin file: " + input.string());
    }
    return ResolveFramesFromFile(input);
  }

  if (!fs::is_directory(input)) {
    throw std::runtime_error("Input path does not exist: " + input.string());
  }

  std::vector<fs::path> files;
  for (const auto & entry : fs::recursive_directory_iterator(input)) {
    if (entry.is_regular_file() && entry.path().extension() == ".bin") {
      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end());
  std::vector<FrameHandle> frames;
  for (const auto & file_path : files) {
    auto resolved = ResolveFramesFromFile(file_path);
    frames.insert(frames.end(), resolved.begin(), resolved.end());
  }

  if (frames.empty()) {
    throw std::runtime_error("No supported .bin frames found in: " + input.string());
  }
  return frames;
}

FrameData LoadFrame(const FrameHandle & handle)
{
  if (handle.format == InputFormat::OusterLegacyContainer) {
    return LoadOusterLegacyFrame(handle);
  }
  return LoadKittiFrame(handle.path);
}

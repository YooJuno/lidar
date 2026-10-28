#include "lidar_viewer/file_dialog.hpp"
#include "lidar_viewer/platform.hpp"
#include <commdlg.h>
#include <atomic>
#include <chrono>
#include <future>
#include <vector>

namespace lidar_viewer
{
struct FileDialog::NativeState
{
  std::atomic<HWND> window{nullptr};
  std::atomic<bool> cancelled{false};
  std::future<FileDialogResult> result;
  static UINT_PTR CALLBACK Hook(HWND child, UINT message, WPARAM, LPARAM parameter)
  {
    if (message==WM_INITDIALOG) {
      auto * state=reinterpret_cast<NativeState *>(reinterpret_cast<OPENFILENAMEW *>(parameter)->lCustData);
      const auto dialog=GetParent(child); state->window.store(dialog);
      if (state->cancelled.load()) PostMessageW(dialog,WM_CLOSE,0,0);
    }
    return 0;
  }
};
FileDialog::~FileDialog() {Cancel();}
bool FileDialog::Active() const {return native_!=nullptr;}
void FileDialog::OpenBag(const std::filesystem::path & folder)
{
  if (Active()) return;
  auto state=std::make_shared<NativeState>();
  state->result=std::async(std::launch::async,[state,folder] {
    const HRESULT initialized=CoInitializeEx(nullptr,COINIT_APARTMENTTHREADED);
    if (FAILED(initialized)) return FileDialogResult{{},"Cannot initialize native file chooser"};
    std::vector<wchar_t> buffer(32768,0);
    const auto initial=folder.wstring();
    OPENFILENAMEW options{}; options.lStructSize=sizeof(options);
    options.lpstrFilter=L"LiDAR data\0metadata.yaml;*.db3;*.mcap;*.bin\0All files\0*.*\0\0";
    options.lpstrFile=buffer.data(); options.nMaxFile=static_cast<DWORD>(buffer.size());
    options.lpstrInitialDir=initial.c_str(); options.lpstrTitle=L"Open LiDAR data";
    options.Flags=OFN_FILEMUSTEXIST|OFN_PATHMUSTEXIST|OFN_NOCHANGEDIR|OFN_EXPLORER|OFN_ENABLEHOOK;
    options.lpfnHook=NativeState::Hook; options.lCustData=reinterpret_cast<LPARAM>(state.get());
    const bool accepted=GetOpenFileNameW(&options)!=0;
    const auto error=accepted ? 0 : CommDlgExtendedError();
    state->window.store(nullptr); CoUninitialize();
    if (accepted) return FileDialogResult{std::filesystem::path(buffer.data()),{}};
    return FileDialogResult{{}, error ? "Native file chooser failed: "+std::to_string(error) : ""};
  });
  native_=std::move(state);
}
std::optional<FileDialogResult> FileDialog::Poll()
{
  if (!native_ || native_->result.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return {};
  auto result=native_->result.get(); native_.reset(); return result;
}
void FileDialog::Cancel() noexcept
{
  if (!native_) return;
  native_->cancelled.store(true);
  if (const auto window=native_->window.load()) PostMessageW(window,WM_CLOSE,0,0);
  try {native_->result.get();} catch (...) {}
  native_.reset();
}
}  // namespace lidar_viewer

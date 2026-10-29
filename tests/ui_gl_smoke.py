#!/usr/bin/env python3
"""Short, private-Xvfb GUI regression checks; never drives the user's desktop.

Run under xvfb-run after sourcing ROS. The viewer remains paused, profile SAVE
is never clicked, and screenshots capture only this test's application window.
"""
import argparse
import ctypes
import ctypes.util
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

from PIL import Image, ImageChops


class X11:
    def __init__(self):
        display_name = os.environ.get("DISPLAY", "")
        try:
            number = int(display_name.rsplit(":", 1)[1].split(".", 1)[0])
        except (IndexError, ValueError) as error:
            raise RuntimeError("Run this test under xvfb-run, not a desktop display") from error
        if not display_name.startswith(":") or number < 90:
            raise RuntimeError("Refusing to drive a desktop DISPLAY; use private xvfb-run")
        self.display_name = display_name
        self.x = ctypes.CDLL(ctypes.util.find_library("X11"))
        self.xt = ctypes.CDLL(ctypes.util.find_library("Xtst"))
        pointer = ctypes.c_void_p
        window = ctypes.c_ulong
        integer = ctypes.c_int
        unsigned = ctypes.c_uint
        self.x.XOpenDisplay.argtypes = [ctypes.c_char_p]
        self.x.XOpenDisplay.restype = pointer
        self.x.XDefaultRootWindow.argtypes = [pointer]
        self.x.XDefaultRootWindow.restype = window
        self.x.XQueryTree.argtypes = [pointer, window, ctypes.POINTER(window),
                                     ctypes.POINTER(window), ctypes.POINTER(ctypes.POINTER(window)),
                                     ctypes.POINTER(unsigned)]
        self.x.XQueryTree.restype = integer
        self.x.XFetchName.argtypes = [pointer, window, ctypes.POINTER(ctypes.c_char_p)]
        self.x.XFetchName.restype = integer
        self.x.XFree.argtypes = [pointer]
        self.x.XGetGeometry.argtypes = [pointer, window, ctypes.POINTER(window),
                                       ctypes.POINTER(integer), ctypes.POINTER(integer),
                                       ctypes.POINTER(unsigned), ctypes.POINTER(unsigned),
                                       ctypes.POINTER(unsigned), ctypes.POINTER(unsigned)]
        self.x.XGetGeometry.restype = integer
        self.x.XTranslateCoordinates.argtypes = [pointer, window, window, integer, integer,
                                                 ctypes.POINTER(integer), ctypes.POINTER(integer),
                                                 ctypes.POINTER(window)]
        self.x.XTranslateCoordinates.restype = integer
        self.x.XMoveResizeWindow.argtypes = [pointer, window, integer, integer, unsigned, unsigned]
        self.x.XSetInputFocus.argtypes = [pointer, window, integer, ctypes.c_ulong]
        self.x.XStringToKeysym.argtypes = [ctypes.c_char_p]
        self.x.XStringToKeysym.restype = ctypes.c_ulong
        self.x.XKeysymToKeycode.argtypes = [pointer, ctypes.c_ulong]
        self.x.XKeysymToKeycode.restype = ctypes.c_ubyte
        self.x.XSync.argtypes = [pointer, integer]
        self.x.XCloseDisplay.argtypes = [pointer]
        self.xt.XTestFakeMotionEvent.argtypes = [pointer, integer, integer, integer, ctypes.c_ulong]
        self.xt.XTestFakeButtonEvent.argtypes = [pointer, unsigned, integer, ctypes.c_ulong]
        self.xt.XTestFakeKeyEvent.argtypes = [pointer, unsigned, integer, ctypes.c_ulong]
        self.display = self.x.XOpenDisplay(display_name.encode())
        if not self.display:
            raise RuntimeError("Could not connect to private Xvfb display")
        self.root = self.x.XDefaultRootWindow(self.display)

    def close(self):
        if self.display:
            self.x.XCloseDisplay(self.display)
            self.display = None

    def title(self, window):
        name = ctypes.c_char_p()
        if not self.x.XFetchName(self.display, window, ctypes.byref(name)) or not name:
            return ""
        try:
            return ctypes.string_at(name).decode(errors="replace")
        finally:
            self.x.XFree(name)

    def children(self, parent):
        root, ancestor = ctypes.c_ulong(), ctypes.c_ulong()
        children = ctypes.POINTER(ctypes.c_ulong)()
        count = ctypes.c_uint()
        if not self.x.XQueryTree(self.display, parent, ctypes.byref(root), ctypes.byref(ancestor),
                                 ctypes.byref(children), ctypes.byref(count)):
            return []
        try:
            return [children[i] for i in range(count.value)]
        finally:
            if children:
                self.x.XFree(children)

    def viewer_window(self):
        for window in self.children(self.root):
            if self.title(window) in ("LiDAR Studio", "LiDAR OpenGL Viewer"):
                return window
        return None

    def geometry(self, window):
        root = ctypes.c_ulong()
        x, y = ctypes.c_int(), ctypes.c_int()
        width, height, border, depth = [ctypes.c_uint() for _ in range(4)]
        if not self.x.XGetGeometry(self.display, window, ctypes.byref(root), ctypes.byref(x),
                                   ctypes.byref(y), ctypes.byref(width), ctypes.byref(height),
                                   ctypes.byref(border), ctypes.byref(depth)):
            raise RuntimeError("Viewer window is no longer available")
        return width.value, height.value

    def resize(self, window, width, height):
        self.x.XMoveResizeWindow(self.display, window, 0, 0, width, height)
        self.x.XSync(self.display, False)
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if self.geometry(window) == (width, height):
                time.sleep(.35)
                return
            time.sleep(.05)
        raise AssertionError(f"Viewer did not resize to {width}x{height}")

    def focus(self, window):
        # GLFW creates its X window before OpenGL initialization finishes.
        # XSetInputFocus on that still-unmapped window aborts Xlib with BadMatch.
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            info = subprocess.run(["xwininfo", "-id", str(window)], capture_output=True,
                                  text=True, timeout=2)
            if "Map State: IsViewable" in info.stdout:
                break
            time.sleep(.05)
        else:
            raise TimeoutError("Viewer window was not mapped before focus")
        self.x.XSetInputFocus(self.display, window, 2, 0)
        self.x.XSync(self.display, False)

    def move(self, window, x, y):
        root_x, root_y, child = ctypes.c_int(), ctypes.c_int(), ctypes.c_ulong()
        self.x.XTranslateCoordinates(self.display, window, self.root, int(x), int(y),
                                     ctypes.byref(root_x), ctypes.byref(root_y), ctypes.byref(child))
        self.xt.XTestFakeMotionEvent(self.display, -1, root_x.value, root_y.value, 0)
        self.x.XSync(self.display, False)
        time.sleep(.12)

    def click(self, window, x, y, button=1):
        self.move(window, x, y)
        self.xt.XTestFakeButtonEvent(self.display, button, True, 0)
        self.xt.XTestFakeButtonEvent(self.display, button, False, 0)
        self.x.XSync(self.display, False)
        time.sleep(.25)

    def wheel(self, window, x, y, count=1):
        self.move(window, x, y)
        for _ in range(abs(count)):
            button = 4 if count > 0 else 5
            self.xt.XTestFakeButtonEvent(self.display, button, True, 0)
            self.xt.XTestFakeButtonEvent(self.display, button, False, 0)
        self.x.XSync(self.display, False)
        time.sleep(.35)

    def key(self, window, keysym):
        self.focus(window)
        code = self.x.XKeysymToKeycode(self.display, self.x.XStringToKeysym(keysym.encode()))
        if not code:
            raise RuntimeError(f"No X11 keycode for {keysym}")
        self.xt.XTestFakeKeyEvent(self.display, code, True, 0)
        self.xt.XTestFakeKeyEvent(self.display, code, False, 0)
        self.x.XSync(self.display, False)
        time.sleep(.35)


def capture(x11, window, destination):
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "x11grab",
        "-draw_mouse", "0", "-window_id", str(window), "-i", x11.display_name,
        "-frames:v", "1", "-n", str(destination)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=5)
    with Image.open(destination) as screenshot:
        return screenshot.convert("RGB")


def changed_pixels(before, after, crop):
    difference = ImageChops.difference(before.crop(crop), after.crop(crop))
    red, green, blue = difference.split()
    any_channel = ImageChops.lighter(ImageChops.lighter(red, green), blue)
    return any_channel.width * any_channel.height - any_channel.histogram()[0]


def settings_inventory(repo):
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((repo / "config").rglob("*")) if path.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--dpi", type=int, default=96)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    viewer, bag = args.viewer.resolve(), args.bag.expanduser().resolve()
    if not viewer.is_file() or not (bag / "metadata.yaml").is_file():
        parser.error("Expected an existing viewer binary and ROS 2 bag directory")
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="gl-smoke-", dir=args.output.resolve()))
    private=output/"private_project"
    (private/"build").mkdir(parents=True)
    shutil.copytree(repo/"config",private/"config")
    (private/"config/profiles/rosbag.cfg").write_text(
        "ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\n"
        "roi_enabled=0\nroi_count=0\nimu_level_enabled=0\nui_scale=1\ncolor_mode=0\n")
    private_viewer=private/"build/lidar_viewer";shutil.copy2(viewer,private_viewer)
    process = None
    x11 = None
    window = None
    before_settings = settings_inventory(repo)
    result = {"status": "started", "output": str(output), "checks": {}, "screenshots": {}}
    result["requested_dpi"] = args.dpi
    started = time.monotonic()
    deadline = started + max(10, min(args.timeout, 90))

    def shot(name):
        if time.monotonic() > deadline:
            raise TimeoutError("GUI smoke test exceeded its time limit")
        path = output / f"{name}.png"
        picture = capture(x11, window, path)
        result["screenshots"][name] = str(path)
        return picture

    try:
        x11 = X11()
        subprocess.run(["xrdb", "-merge"], input=f"Xft.dpi: {args.dpi}\n", text=True, check=True)
        with (output / "viewer.log").open("wb") as log:
            process = subprocess.Popen(
                [str(private_viewer), str(bag), "--topic", "/b2/points", "--sensor-height", "0.57"],
                cwd=private, env=dict(os.environ,XDG_CONFIG_HOME=str(output/"isolated_config")),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while not window:
                if process.poll() is not None:
                    raise RuntimeError(f"Viewer exited before creating a window: {process.returncode}")
                if time.monotonic() > min(deadline, started + 40):
                    raise TimeoutError("Viewer did not finish indexing/opening within 40 seconds")
                window = x11.viewer_window()
                if not window:
                    time.sleep(.1)
            x11.focus(window)
            x11.resize(window, 1600, 960)
            while "Opened bag:" not in (output / "viewer.log").read_text(errors="replace"):
                if process.poll() is not None or time.monotonic()>deadline: raise TimeoutError("Initial bag did not load")
                time.sleep(.1)
            x11.move(window, 150, 350)
            time.sleep(.7)
            original = shot("01_desktop_1600")
            ui_scale = max(.75, min(args.dpi / 96., 1600 / 1024., 960 / 720., 2.))
            crop = (int(380 * ui_scale), int(120 * ui_scale), 1560,
                    int(960 - 28 * ui_scale - 16 * ui_scale - 110 * ui_scale - 32 * ui_scale))
            x11.wheel(window, 150, 350, 2)
            sidebar_wheel = shot("02_sidebar_wheel")
            changed = changed_pixels(original, sidebar_wheel, crop)
            result["checks"]["wheel_over_ui_canvas_changed_pixels"] = changed
            if changed != 0:
                raise AssertionError(f"Wheel over sidebar changed {changed} canvas pixels")
            x11.wheel(window, 900, 400, 2)
            zoomed = shot("03_scene_zoom")
            changed = changed_pixels(sidebar_wheel, zoomed, crop)
            result["checks"]["wheel_over_scene_changed_pixels"] = changed
            if changed < 50:
                raise AssertionError("Wheel over canvas did not visibly zoom the scene")
            x11.click(window, 900, 400)
            x11.key(window, "c")
            uniform = shot("04_uniform_color")
            changed = changed_pixels(zoomed, uniform, crop)
            result["checks"]["color_shortcut_changed_pixels"] = changed
            if changed < 50:
                raise AssertionError("Scene color shortcut did not change point colors")

            for width, height in [(1024, 640), (800, 600)]:
                x11.resize(window, width, height)
                picture = shot(f"05_compact_{width}x{height}")
                if picture.size != (width, height):
                    raise AssertionError("Captured window size differs from requested compact size")
                if process.poll() is not None:
                    raise RuntimeError("Viewer exited after resizing")
                result["checks"][f"resize_{width}x{height}"] = True
            x11.resize(window, 1600, 960)
            x11.click(window, 155 * ui_scale, 141 * ui_scale)
            shot("06_processing_1600")
            result["checks"]["settings_unchanged"] = before_settings == settings_inventory(repo)
            if not result["checks"]["settings_unchanged"]:
                raise AssertionError("A GUI smoke test unexpectedly changed saved settings")
            result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
    finally:
        if process is not None and process.poll() is None:
            try:
                if x11 and window:
                    x11.click(window, 900, 400)
                    x11.key(window, "Escape")
                process.wait(timeout=2)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        if x11:
            x11.close()
        result["elapsed_s"] = time.monotonic() - started
        result["viewer_exit_code"] = process.returncode if process else None
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

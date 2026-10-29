#!/usr/bin/env python3
"""Exercise empty startup and button-driven GTK bag selection on private Xvfb."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import traceback

from ui_gl_smoke import X11, capture, settings_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    viewer, bag, target = [p.expanduser().resolve() for p in (args.viewer, args.bag, args.target)]
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="file-dialog-", dir=args.output.resolve()))
    env = dict(os.environ, XDG_CONFIG_HOME=str(output / "isolated_config"),
        GSETTINGS_BACKEND="memory", GTK_USE_PORTAL="0", NO_AT_BRIDGE="1")
    # A registered recording must populate the library without being opened.
    library = Path(env["XDG_CONFIG_HOME"]) / "lidar_viewer/data_folders.cfg"
    library.parent.mkdir(parents=True)
    library_text = 'folder "' + str(bag) + '"\n'
    library.write_text(library_text)
    before = settings_inventory(repo)
    result = {"status": "started", "checks": {}, "output": str(output)}
    process = x11 = window = None
    children = {}
    deadline = time.monotonic() + 85
    log_path = output / "viewer.log"

    def alive():
        if time.monotonic() > deadline: raise TimeoutError("File chooser test timed out")
        if process.poll() is not None: raise RuntimeError(f"Viewer exited: {process.returncode}")
        try:
            for pid in Path(f"/proc/{process.pid}/task/{process.pid}/children").read_text().split():
                stat = Path(f"/proc/{pid}/stat").read_text().split()
                children[int(pid)] = stat[21]
        except FileNotFoundError: pass

    def wait_for(predicate):
        while not predicate():
            alive()
            time.sleep(.1)

    def chooser_window():
        return next((w for w in x11.children(x11.root) if x11.title(w) == "Open LiDAR data"), None)

    def event(key, down):
        code = x11.x.XKeysymToKeycode(x11.display, x11.x.XStringToKeysym(key.encode()))
        if not code: raise RuntimeError(f"No key for {key}")
        x11.xt.XTestFakeKeyEvent(x11.display, code, down, 0)

    def chord(window, key):
        x11.focus(window)
        event("Control_L", True); event(key, True); event(key, False); event("Control_L", False)
        x11.x.XSync(x11.display, False)
        time.sleep(.3)

    def close_viewer():
        # Send the window manager's close action; Escape belongs to whichever
        # editor/dialog currently holds keyboard focus.
        class Data(ctypes.Union):
            _fields_ = [("l", ctypes.c_long * 5), ("b", ctypes.c_char * 20)]
        class Message(ctypes.Structure):
            _fields_ = [("type", ctypes.c_int), ("serial", ctypes.c_ulong),
                ("send_event", ctypes.c_int), ("display", ctypes.c_void_p),
                ("window", ctypes.c_ulong), ("message_type", ctypes.c_ulong),
                ("format", ctypes.c_int), ("data", Data)]
        class Event(ctypes.Union):
            _fields_ = [("message", Message), ("pad", ctypes.c_long * 24)]
        x11.x.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        x11.x.XInternAtom.restype = ctypes.c_ulong
        x11.x.XSendEvent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_int,
            ctypes.c_long, ctypes.POINTER(Event)]
        event = Event()
        event.message.type = 33
        event.message.display = x11.display
        event.message.window = window
        event.message.message_type = x11.x.XInternAtom(x11.display, b"WM_PROTOCOLS", False)
        event.message.format = 32
        event.message.data.l[0] = x11.x.XInternAtom(x11.display, b"WM_DELETE_WINDOW", False)
        x11.x.XSendEvent(x11.display, window, False, 0, ctypes.byref(event))
        x11.x.XSync(x11.display, False)

    def choose(path, name):
        wait_for(chooser_window)
        dialog = chooser_window()
        x11.focus(dialog)
        capture(x11, dialog, output / f"{name}.png")
        chord(dialog, "l")
        event("Control_L", True); event("a", True); event("a", False); event("Control_L", False)
        for ch in str(path):
            shifted = ch.isupper() or ch == "_"
            key = {"/": "slash", "_": "underscore", ".": "period", "-": "minus", " ": "space"}.get(ch, ch.lower())
            if shifted: event("Shift_L", True)
            event(key, True); event(key, False)
            if shifted: event("Shift_L", False)
        x11.x.XSync(x11.display, False)
        time.sleep(.3)
        event("Return", True); event("Return", False)
        x11.x.XSync(x11.display, False)
        time.sleep(.8)
        if chooser_window():
            x11.focus(chooser_window())
            x11.key(chooser_window(), "Return")
        wait_for(lambda: not chooser_window())

    try:
        x11 = X11()
        subprocess.run(["xrdb", "-merge"], input="Xft.dpi: 96\n", text=True, check=True)
        with log_path.open("wb") as log:
            # No input argument: show the viewer, with no dialog or data loaded.
            process = subprocess.Popen([str(viewer), "--2d", "--imu-level", "--play"], cwd=repo,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            wait_for(x11.viewer_window)
            window = x11.viewer_window()
            x11.focus(window); x11.resize(window, 1600, 960)
            time.sleep(.4)
            initial_log = log_path.read_text(errors="replace")
            result["checks"]["viewer_opens_before_file_chooser"] = chooser_window() is None
            result["checks"]["registered_bag_not_loaded_automatically"] = (
                "No bag loaded." in initial_log and "Indexing ROS" not in initial_log and "Input:" not in initial_log)
            capture(x11, window, output / "00_empty_viewer.png")
            # Empty playback/seek controls and shortcuts must be safe.
            for point in [(410, 837), (480, 837), (525, 837), (1050, 878)]:
                x11.click(window, *point)
            x11.key(window, "space"); x11.key(window, "n"); x11.key(window, "p")
            alive()
            result["checks"]["empty_playback_controls_are_safe"] = "Indexing ROS" not in log_path.read_text(errors="replace")
            x11.click(window, 240, 30)
            wait_for(chooser_window)
            x11.key(chooser_window(), "Escape")
            wait_for(lambda: not chooser_window())
            time.sleep(.3)
            alive()
            result["checks"]["cancel_from_empty_keeps_viewer_open"] = "Opened bag:" not in log_path.read_text(errors="replace")
            result["stage"] = "first_bag_selection"
            x11.focus(window)
            x11.click(window, 240, 30)
            choose(bag / "metadata.yaml", "01_button_chooser")
            wait_for(lambda: f'Opened bag: "{bag}"' in log_path.read_text(errors="replace"))
            capture(x11, window, output / "02_selected_bag.png")
            result["checks"]["selected_metadata_opens_bag"] = bag.name in log_path.read_text(errors="replace")
            # The Open file button opens the system chooser; cancellation is benign.
            x11.click(window, 240, 30)
            result["stage"] = "runtime_cancel"
            wait_for(chooser_window)
            x11.key(chooser_window(), "Escape")
            wait_for(lambda: not chooser_window())
            time.sleep(.3)
            alive()
            result["checks"]["cancel_keeps_current_bag"] = log_path.read_text(errors="replace").count("Opened bag:") == 1
            chord(window, "o")
            result["stage"] = "runtime_selection"
            chunk = next(iter(sorted(target.glob("*.mcap"))), None) or next(iter(sorted(target.glob("*.db3"))))
            choose(chunk, "03_runtime_chooser")
            wait_for(lambda: f'Opened bag: "{target}"' in log_path.read_text(errors="replace"))
            capture(x11, window, output / "04_other_bag.png")
            result["checks"]["ctrl_o_opens_complete_split_recording"] = True
            # Closing the viewer must also close a pending native chooser.
            chord(window, "o")
            result["stage"] = "close_pending_chooser"
            wait_for(chooser_window)
            alive()
            close_viewer()
            process.wait(timeout=5)
            time.sleep(.2)
            result["checks"]["viewer_exit_closes_chooser"] = process.returncode == 0 and chooser_window() is None
            result["checks"]["profiles_unchanged"] = before == settings_inventory(repo)
            result["checks"]["selection_does_not_register_sibling_folders"] = library.read_text() == library_text
            if not all(result["checks"].values()): raise AssertionError("File chooser checks failed")
            result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if x11:
            try:
                active = chooser_window() or window
                if active: capture(x11, active, output / "failure.png")
            except Exception: pass
    finally:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=3)
        for child, started in children.items():
            try:
                if Path(f"/proc/{child}/stat").read_text().split()[21] == started:
                    os.kill(child, signal.SIGKILL)
            except (ProcessLookupError, FileNotFoundError): pass
        if x11: x11.close()
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

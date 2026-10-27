#!/usr/bin/env python3
"""Private-Xvfb checks for yaw ROI editing, union, restore and profile persistence."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import time
import traceback

from ui_gl_smoke import X11, capture, settings_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="roi-", dir=args.output.resolve()))
    private = output / "private_project"
    (private / "build").mkdir(parents=True)
    shutil.copytree(repo / "config", private / "config")
    viewer = private / "build/lidar_viewer"
    shutil.copy2(args.viewer.resolve(), viewer)
    profile = private / "config/profiles/default.cfg"
    # Ensure a deterministic fixture, independent of the user's preferences.
    profile.write_text("profile=default\nground_removal_enabled=0\nclustering_enabled=0\n"
        "detection_enabled=0\ncalibration_yaw_deg=0\ncalibration_pitch_deg=0\n"
        "calibration_roll_deg=0\ncalibration_z_m=0\n")
    fixture = output / "points.bin"
    with fixture.open("wb") as stream:
        for x in range(-12, 13):
            for y in range(-12, 13):
                for z in range(-3, 2):
                    stream.write(struct.pack("ffff", x, y, z, .5))
    env = dict(os.environ, XDG_CONFIG_HOME=str(output / "isolated_config"))
    before = settings_inventory(repo)
    process = x11 = window = None
    result = {"status": "started", "checks": {}, "output": str(output)}

    def key_event(key, down):
        code = x11.x.XKeysymToKeycode(x11.display, x11.x.XStringToKeysym(key.encode()))
        x11.xt.XTestFakeKeyEvent(x11.display, code, down, 0)

    def enter(x, y, value, ctrl_click=True):
        if ctrl_click: key_event("Control_L", True)
        x11.click(window, x, y)
        key_event("Control_L", True)
        key_event("a", True); key_event("a", False)
        key_event("Control_L", False)
        for ch in str(value):
            key = {"-": "minus", ".": "period"}.get(ch, ch)
            key_event(key, True); key_event(key, False)
        key_event("Return", True); key_event("Return", False)
        x11.x.XSync(x11.display, False)
        time.sleep(.2)

    def count(expected, label, boxes=None):
        x11.click(window, 170, 850)
        inspected = subprocess.run([str(viewer), str(fixture), "--inspect"], cwd=private,
            env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
        match = re.search(r"ROI: .*\| (\d+) boxes \| retained (\d+) / 3125 points", inspected.stdout)
        result["checks"][label] = (inspected.returncode == 0 and match is not None and
            int(match[2]) == expected and (boxes is None or int(match[1]) == boxes))
        if not result["checks"][label]:
            raise AssertionError(f"{label}: expected {expected} points\n{inspected.stdout}")

    try:
        x11 = X11()
        subprocess.run(["xrdb", "-merge"], input="Xft.dpi: 96\n", text=True, check=True)
        with (output / "viewer.log").open("wb") as log:
            process = subprocess.Popen([str(viewer), str(fixture), "--2d"], cwd=private,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + 20
            while not window:
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Viewer did not open")
                window = x11.viewer_window()
                time.sleep(.1)
            x11.focus(window); x11.resize(window, 1600, 960)
            x11.click(window, 280, 141)
            x11.click(window, 75, 380)  # first box
            count(2205, "default_box_and_raw_Z_alignment")
            user_profile=Path(env["XDG_CONFIG_HOME"])/"lidar_viewer/profiles/default.cfg"
            if user_profile.is_file(): profile=user_profile
            enter(260, 646, -2)
            for x, value in [(75, 4), (170, 2), (260, 2)]: enter(x, 714, value)
            enter(170, 756, 90)
            count(45, "yaw_and_full_size_edit")
            enter(267, 756, 45, ctrl_click=False)
            count(21, "direct_yaw_input_without_ctrl_click")
            enter(267, 756, 90, ctrl_click=False)
            count(45, "direct_yaw_input_updates_geometry")
            capture(x11, window, output / "01_yaw_roi.png")
            x11.click(window, 75, 397)  # second box
            enter(75, 646, 8); enter(260, 646, -2)
            for x, value in [(75, 4), (170, 2), (260, 2)]: enter(x, 714, value)
            count(90, "multiple_boxes_union")
            capture(x11, window, output / "02_union.png")
            x11.click(window, 46, 227)
            count(3125, "master_off_restores_full_cloud")
            x11.click(window, 46, 227)
            count(90, "master_on_restores_union")
            x11.click(window, 46, 537)
            count(45, "individual_box_disabled")
            # Both boxes disabled must give an empty crop, with no fall-through.
            x11.click(window, 100, 437)
            x11.click(window, 46, 537)
            count(0, "all_boxes_disabled_is_empty")
            capture(x11, window, output / "03_empty_roi.png")
            # Restore and exercise delete/duplicate controls after scrolling.
            x11.wheel(window, 250, 700, -5)
            x11.click(window, 46, 432)
            count(45, "restore_one_box")
            x11.click(window, 75, 760)
            count(45, "duplicate_does_not_duplicate_points", 3)
            x11.click(window, 166, 760)
            count(45, "delete_duplicate", 2)
            x11.click(window, 166, 760)
            count(45, "delete_disabled_box", 1)
            capture(x11, window, output / "04_scrolled_editor.png")
            x11.click(window, 166, 760)
            count(0, "delete_last_box_and_restore_empty_list", 0)
            # Exercise the actual pipeline with nonzero dataset Z alignment and
            # all three display rotations, both before and after segmentation.
            base = profile.read_text()
            base = '\n'.join(line for line in base.splitlines() if not line.startswith('roi_')) + '\n'
            base += ('roi_enabled=1\nroi_count=2\nroi_box=1 "a" 0 0 -2 4 2 2 90\n'
                'roi_box=1 "b" 8 0 -2 4 2 2 0\ncalibration_yaw_deg=19\n'
                'calibration_pitch_deg=-12\ncalibration_roll_deg=9\ncalibration_z_m=0.57\n')
            for ground in [0, 1]:
                profile.write_text(base + f'ground_removal_enabled={ground}\nground_mode=fast_grid\n')
                inspected = subprocess.run([str(viewer), str(fixture), "--inspect"], cwd=private,
                    env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=15)
                label = f"pipeline_transform_ground_{ground}"
                result["checks"][label] = (inspected.returncode == 0 and
                    "2 boxes | retained 90 / 3125 points" in inspected.stdout)
                if not result["checks"][label]: raise AssertionError(label + '\n' + inspected.stdout)
            result["checks"]["original_profiles_unchanged"] = before == settings_inventory(repo)
            result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try: capture(x11, window, output / "failure.png")
            except Exception: pass
    finally:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=3)
            except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=3)
        if x11: x11.close()
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

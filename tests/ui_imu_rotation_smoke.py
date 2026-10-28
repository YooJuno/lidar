#!/usr/bin/env python3
"""Private-Xvfb checks for 2D rotation, IMU leveling and saved preferences."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback

from ui_gl_smoke import X11, capture, changed_pixels, settings_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roi", action="store_true", help="Also exercise two yaw ROI boxes with IMU leveling")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="imu-rotation-", dir=args.output.resolve()))
    private = output / "private_project"
    (private / "build").mkdir(parents=True)
    shutil.copytree(repo / "config", private / "config")
    profile=private/"config/profiles/rosbag.cfg"
    # User ROI and camera preferences must not alter the test fixture.
    text='\n'.join(line for line in profile.read_text().splitlines()
        if not line.startswith(("roi_","camera_","projection_mode=")))
    text+='\nprojection_mode=3d\ncamera_yaw_rad=-0.8\ncamera_pitch_rad=0.55\ncamera_distance=30\n'
    if args.roi:
        text+='roi_enabled=1\nroi_show_boxes=1\nroi_count=2\n'+(
            'roi_box=1 "Front area" 5 0 -0.5 12 9 4 25\n'
            'roi_box=1 "Rear area" -5 0 -0.5 5 6 4 -15\n')
    profile.write_text(text)
    viewer = private / "build/lidar_viewer"
    shutil.copy2(args.viewer.resolve(), viewer)
    bag = args.bag.expanduser().resolve()
    env = dict(os.environ, XDG_CONFIG_HOME=str(output / "isolated_config"))
    before = settings_inventory(repo)
    process = x11 = window = None
    result = {"status": "started", "checks": {}, "screenshots": {}, "output": str(output)}
    deadline = time.monotonic() + 75
    log_path = output / "viewer.log"
    crop = (380, 120, 1560, 770)

    def alive():
        if time.monotonic() > deadline:
            raise TimeoutError("IMU/2D regression exceeded its deadline")
        if process.poll() is not None:
            raise RuntimeError(f"Viewer exited: {process.returncode}")

    def wait_for(predicate):
        while not predicate():
            alive()
            time.sleep(.1)

    def shot(name):
        alive()
        path = output / (name + ".png")
        result["screenshots"][name] = str(path)
        return capture(x11, window, path)

    def drag(x0, y0, x1, y1, button):
        x11.move(window, x0, y0)
        x11.xt.XTestFakeButtonEvent(x11.display, button, True, 0)
        x11.x.XSync(x11.display, False)
        for i in range(1, 6):
            x11.move(window, x0 + (x1-x0)*i/5, y0 + (y1-y0)*i/5)
        x11.xt.XTestFakeButtonEvent(x11.display, button, False, 0)
        x11.x.XSync(x11.display, False)
        time.sleep(.3)

    try:
        x11 = X11()
        subprocess.run(["xrdb", "-merge"], input="Xft.dpi: 96\n", text=True, check=True)
        with log_path.open("wb") as log:
            process = subprocess.Popen([str(viewer), str(bag), "--2d", "--imu-level",
                "--sensor-height", ".57"], cwd=private, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            while not window:
                alive()
                window = x11.viewer_window()
                if not window:
                    time.sleep(.1)
            x11.focus(window)
            x11.resize(window, 1600, 960)
            time.sleep(.4)
            original = shot("01_top_view")
            x11.click(window, 900, 450)
            for _ in range(4):
                x11.key(window, "e")
            rotated = shot("02_keyboard_rotation")
            result["checks"]["keyboard_rotates_2d"] = changed_pixels(original, rotated, crop) > 100
            drag(900, 450, 1020, 450, 3)
            mouse_rotated = shot("03_mouse_rotation")
            result["checks"]["right_drag_rotates_2d"] = changed_pixels(rotated, mouse_rotated, crop) > 100
            drag(900, 450, 1000, 490, 1)
            panned = shot("04_rotated_pan")
            result["checks"]["left_drag_pans_rotated_view"] = changed_pixels(mouse_rotated, panned, crop) > 100
            # Seek beyond the beginning of the bag where IMU coverage starts.
            x11.click(window, 1000, 878)
            wait_for(lambda: "IMU leveling active:" in log_path.read_text(errors="replace"))
            x11.click(window, 163, 141)
            time.sleep(.3)
            leveled = shot("05_imu_leveling_active")
            x11.click(window, 46, 222)
            unlevel = shot("06_imu_leveling_off")
            result["checks"]["imu_toggle_changes_geometry"] = changed_pixels(leveled, unlevel, crop) > 50
            x11.click(window, 46, 222)
            wait_for(lambda: log_path.read_text(errors="replace").count("IMU leveling active:") >= 2)
            shot("07_imu_leveling_restored")
            # This binary copy resolves profiles inside private_project.
            x11.click(window, 170, 850)
            saved = private / "config/profiles/rosbag.cfg"
            wait_for(lambda: "imu_level_enabled=1" in saved.read_text())
            values = dict(line.split("=", 1) for line in saved.read_text().splitlines() if "=" in line)
            result["checks"]["saved_2d_rotation"] = (values["projection_mode"] == "2d" and
                abs(float(values["camera_2d_yaw_rad"])) > .1)
            result["checks"]["preserved_3d_view"] = abs(float(values["camera_3d_yaw_rad"]) + .8) < 1e-5
            result["checks"]["saved_imu_preference"] = values["imu_level_enabled"] == "1"
            if args.roi:
                result["checks"]["saved_roi_with_imu"] = values["roi_enabled"] == "1" and values["roi_count"] == "2"
            restarted = subprocess.run([str(viewer), str(bag), "--inspect"], cwd=private, env=env,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
            (output / "restart_inspect.log").write_text(restarted.stdout)
            result["checks"]["restored_imu_preference"] = (restarted.returncode == 0 and
                "IMU leveling:" in restarted.stdout)
            result["checks"]["original_profiles_unchanged"] = before == settings_inventory(repo)
            if not all(result["checks"].values()):
                raise AssertionError("One or more IMU/rotation regression checks failed")
            result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try:
                shot("failure")
            except Exception:
                pass
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        if x11:
            x11.close()
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

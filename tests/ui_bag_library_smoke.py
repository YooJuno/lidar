#!/usr/bin/env python3
"""Exercise explicit folder registration and bag switching on private Xvfb.

The user's desktop and folder library are never touched. Run after sourcing ROS:
  xvfb-run -a -s '-screen 0 1920x1200x24' python3 tests/ui_bag_library_smoke.py \
    --viewer build/lidar_viewer --bag /path/to/bag --folder /path/to/bags \
    --target /path/to/another_bag --output build/test_artifacts/bag_library
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import traceback

from ui_gl_smoke import X11, capture, changed_pixels, settings_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=80)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    viewer, bag, folder, target = [path.expanduser().resolve() for path in
                                    (args.viewer, args.bag, args.folder, args.target)]
    if not viewer.is_file() or not all((path / "metadata.yaml").is_file() for path in (bag, target)):
        parser.error("Expected viewer binary and two recorded ROS 2 bag folders")
    if not folder.is_dir() or folder != bag.parent or target.parent != folder:
        parser.error("This regression fixture expects both bags immediately inside --folder")
    # The search value contains only digits, avoiding keyboard-layout-specific
    # punctuation. Sensor recording names use YYYYMMDD_HHMMSS.
    search = target.name.split("_")[-2]
    if not search.isdigit():
        parser.error("Target recording name must end in _HHMMSS_suffix")
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="bag-library-", dir=args.output.resolve()))
    env = os.environ.copy()
    env["XDG_CONFIG_HOME"] = str(output / "isolated_config")
    library = Path(env["XDG_CONFIG_HOME"]) / "lidar_viewer/data_folders.cfg"
    before_settings = settings_inventory(repo)
    process = x11 = window = None
    result = {"status": "started", "output": str(output), "checks": {}, "screenshots": {}}
    started = time.monotonic()
    deadline = started + max(15, min(args.timeout, 90))
    log_path = output / "viewer.log"

    def check_time():
        if time.monotonic() > deadline:
            raise TimeoutError("Bag library GUI smoke test exceeded its deadline")
        if process and process.poll() is not None:
            raise RuntimeError(f"Viewer exited unexpectedly: {process.returncode}")

    def shot(name):
        check_time()
        path = output / f"{name}.png"
        picture = capture(x11, window, path)
        result["screenshots"][name] = str(path)
        return picture

    def wait_until(predicate, description):
        while not predicate():
            check_time()
            time.sleep(.1)
        result["checks"][description] = True

    try:
        x11 = X11()  # Refuses physical DISPLAYs before sending any input.
        subprocess.run(["xrdb", "-merge"], input="Xft.dpi: 96\n", text=True, check=True)
        with log_path.open("wb") as log:
            process = subprocess.Popen([str(viewer), str(bag), "--topic", "/b2/points",
                                        "--sensor-height", "0.57"], cwd=repo, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while not window:
                check_time()
                window = x11.viewer_window()
                if not window:
                    time.sleep(.1)
            x11.focus(window)
            x11.resize(window, 1600, 960)
            time.sleep(.6)
            original = shot("01_initial_single_bag")
            initial_log = log_path.read_text(errors="replace")
            if "Bag list: 1 recordings" not in initial_log:
                raise AssertionError("An explicit bag unexpectedly discovered sibling data")
            result["checks"]["explicit_bag_starts_with_one_source"] = True
            x11.click(window, 540, 872)
            shot("02_add_folder_picker")
            # The picker initially browses the current recording's parent.
            # 700x560 centered modal at (450,200), with its footer below the
            # directory child. Click the explicit Add this folder action.
            x11.click(window, 535, 678)
            wait_until(library.is_file, "folder_library_saved")
            if str(folder) not in library.read_text():
                raise AssertionError("Folder library did not persist the chosen source")
            time.sleep(.3)
            shot("03_registered_folder")
            x11.click(window, 1120, 837)
            shot("04_bag_list")
            # Source popup contains a date selector followed by a name search.
            # Click the name field and type target's numeric acquisition time.
            x11.click(window, 1210, 474)
            for digit in search:
                x11.key(window, digit)
            shot("05_filtered_bag_list")
            x11.click(window, 1120, 761)
            wait_until(lambda: f"Opened bag: \"{target}\"" in log_path.read_text(errors="replace") or
                       f"Opened bag: {target}" in log_path.read_text(errors="replace"),
                       "other_bag_opened")
            time.sleep(.5)
            replacement = shot("06_other_bag_first_frame")
            changed = changed_pixels(original, replacement, (390, 110, 1560, 770))
            result["checks"]["canvas_changed_pixels"] = changed
            if changed < 50:
                raise AssertionError("Source changed in log but point cloud canvas did not change")
            # A fresh no-argument process must recover the explicitly saved
            # folder catalog and open its newest bag without any UI interaction.
            inspect = subprocess.run([str(viewer), "--inspect", "--topic", "/b2/points"],
                                     cwd=repo, env=env, text=True, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,
                                     timeout=max(1, min(20, deadline - time.monotonic())))
            (output / "restart_inspect.log").write_text(inspect.stdout)
            if inspect.returncode or target.name not in inspect.stdout or bag.name not in inspect.stdout:
                raise AssertionError("Saved folder library was not restored on the next launch")
            result["checks"]["library_restored_without_input"] = True
            result["checks"]["settings_unchanged"] = before_settings == settings_inventory(repo)
            if not result["checks"]["settings_unchanged"]:
                raise AssertionError("Library test changed a dataset profile")
            result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try:
                path = output / "failure.png"
                capture(x11, window, path)
                result["screenshots"]["failure"] = str(path)
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
        result["elapsed_s"] = time.monotonic() - started
        result["viewer_exit_code"] = process.returncode if process else None
        (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

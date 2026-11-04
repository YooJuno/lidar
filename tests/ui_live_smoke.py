#!/usr/bin/env python3
"""Private ROS domain + Xvfb: verify live motion, frozen display, resume and raw recording."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
import yaml
import numpy as np

from ui_gl_smoke import X11, capture, changed_pixels, settings_inventory
from ui_controls import Controls


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--connect-with-dialog", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="live-ui-", dir=args.output.resolve()))
    private = output / "private_project"
    (private / "build").mkdir(parents=True)
    shutil.copytree(repo / "config", private / "config")
    profile = private / "config/profiles/rosbag.cfg"
    clean_profile = '\n'.join(line for line in profile.read_text().splitlines() if not line.startswith(('roi_','display_roi_')))
    profile.write_text(clean_profile + "\nground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\n"
        "roi_enabled=0\nroi_count=0\ndisplay_roi_enabled=0\ndisplay_roi_count=1\n"
        'display_roi_box=1 "export" 0 0 0 100 100 100 0\n')
    viewer = private / "build/lidar_viewer"
    shutil.copy2(args.viewer.resolve(), viewer)
    env = dict(os.environ, ROS_DOMAIN_ID="188", ROS_AUTOMATIC_DISCOVERY_RANGE="LOCALHOST",
        XDG_CONFIG_HOME=str(output / "isolated_config"),LIDAR_VIEWER_UI_TRACE="1")
    before = settings_inventory(repo)
    result = {"status": "started", "checks": {}, "output": str(output)}
    process = publisher = x11 = window = None
    deadline = time.monotonic() + 65
    log_path = output / "viewer.log"
    crop = (390, 180, 1540, 750)

    def wait_for(predicate):
        while not predicate():
            if process.poll() is not None: raise RuntimeError("Viewer exited unexpectedly")
            if publisher.poll() is not None: raise RuntimeError("Fixture publisher exited")
            if time.monotonic() > deadline: raise TimeoutError("Live GUI test timed out")
            time.sleep(.1)

    def shot(name):
        return capture(x11, window, output / (name + ".png"))

    try:
        x11 = X11()
        subprocess.run(["xrdb", "-merge"], input="Xft.dpi: 96\n", text=True, check=True)
        with (output / "publisher.log").open("wb") as publog, log_path.open("wb") as log:
            publisher = subprocess.Popen(["/usr/bin/python3", str(repo / "tests/live_fixture.py")],
                env=env, stdout=publog, stderr=subprocess.STDOUT)
            live_args=[] if args.connect_with_dialog else ["--live"]
            process = subprocess.Popen([str(viewer), *live_args, "--topic", "/fixture/points",
                "--imu-topic", "/fixture/imu", "--2d", "--imu-level", "--sensor-height", ".57",
                "--record-folder", str(output / "recordings")], cwd=private, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            wait_for(x11.viewer_window)
            window = x11.viewer_window()
            x11.focus(window); x11.resize(window, 1600, 960)
            if args.connect_with_dialog:
                x11.click(window,350,30)
                shot("00_live_dialog")
                x11.click(window,555,638)
            wait_for(lambda: "Live first frame:" in log_path.read_text(errors="replace"))
            time.sleep(.3)
            moving_a = shot("01_live")
            time.sleep(.35)
            moving_b = shot("02_live_motion")
            result["checks"]["live_cloud_moves"] = changed_pixels(moving_a, moving_b, crop) > 80
            # Tabs, edit-tool activation and common settings must leave the stream running.
            for x in [50,100,155,255]: x11.click(window,x,140)
            x11.click(window,205,140);x11.click(window,65,290)
            x11.click(window,1567,30);x11.key(window,"Escape")
            active_a=shot("02a_after_controls");time.sleep(.35);active_b=shot("02b_controls_still_live")
            result["checks"]["ordinary_controls_do_not_pause_live"]=changed_pixels(active_a,active_b,crop)>80 and "Live display paused" not in log_path.read_text(errors="replace")
            x11.click(window, 410, 837)
            wait_for(lambda: "Live display paused" in log_path.read_text(errors="replace"))
            frozen_a = shot("03_frozen")
            time.sleep(.4)
            frozen_b = shot("04_still_frozen")
            result["checks"]["pause_freezes_geometry"] = changed_pixels(frozen_a, frozen_b, crop) == 0
            # Recording is independent of the paused display.
            x11.click(window, 420, 878)
            wait_for(lambda: "Live recording:" in log_path.read_text(errors="replace"))
            time.sleep(.8)
            frozen_recording = shot("05_recording_while_frozen")
            result["checks"]["record_does_not_unfreeze_display"] = changed_pixels(frozen_a, frozen_recording, crop) == 0
            x11.click(window, 420, 878)
            wait_for(lambda: len(list((output / "recordings").glob("*/metadata.yaml"))) == 1)
            metadata = next((output / "recordings").glob("*/metadata.yaml"))
            data = yaml.safe_load(metadata.read_text())["rosbag2_bagfile_information"]
            counts = {t["topic_metadata"]["name"]: t["message_count"] for t in data["topics_with_message_count"]}
            result["checks"]["received_and_recorded_while_frozen"] = counts.get("/fixture/points", 0) >= 8 and counts.get("/fixture/imu", 0) >= 50
            result["checks"]["static_tf_recorded"] = counts.get("/tf_static", 0) >= 1
            # Export the exact frozen frame, using raw XYZI rather than display
            # colors or a new frame pulled from the still-running subscriber.
            controls=Controls(x11,window,log_path)
            controls.click('tab_ROI');controls.click('roi_purpose_display');controls.click('roi_export')
            controls.enter('export_directory','.')
            controls.click('export_start')
            wait_for(lambda:len(list(private.glob('live-*_seq*/metadata.json')))==1 or
                'ROI export failed:' in log_path.read_text(errors='replace'))
            if 'ROI export failed:' in log_path.read_text(errors='replace'):raise AssertionError('Live ROI export failed')
            exported=next(private.glob('live-*_seq*/metadata.json'))
            wait_for(lambda:json.loads(exported.read_text())['status']=='complete')
            dataset=json.loads(exported.read_text());sample=dataset['samples'][0]
            detail=json.loads((exported.parent/sample['metadata']).read_text())
            cloud=np.load(exported.parent/sample['points'],allow_pickle=False)
            result["checks"]["frozen_live_roi_export_has_source_and_imu"] = (
                detail['source']['kind']=='live' and detail['source']['topic']=='/fixture/points'
                and detail['source']['sequence']==int(re.findall(r'Live display paused at frame (\d+)',log_path.read_text())[-1])
                and detail['imu']['applied'])
            result["checks"]["frozen_live_export_preserves_intensity_and_pause"] = (
                cloud.shape[1]==4 and len(cloud)>0 and np.array_equal(np.unique(cloud[:,3]),[20,70,80,90,100,110,120,130,140])
                and "Live display resumed" not in log_path.read_text(errors="replace"))
            controls.click('export_close')
            x11.click(window, 410, 837)
            wait_for(lambda: "Live display resumed" in log_path.read_text(errors="replace"))
            time.sleep(.3)
            resumed = shot("06_resumed")
            result["checks"]["play_returns_to_live_motion"] = changed_pixels(frozen_a, resumed, crop) > 80
            result["checks"]["live_imu_leveling_active"] = "IMU leveling active:" in log_path.read_text(errors="replace")
            result["checks"]["original_profiles_unchanged"] = before == settings_inventory(repo)
            if not all(result["checks"].values()): raise AssertionError("Live GUI checks failed")
            result["status"] = "passed"
    except Exception as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try: shot("failure")
            except Exception: pass
    finally:
        for child in [process, publisher]:
            if child and child.poll() is None:
                child.terminate()
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired: child.kill(); child.wait(timeout=3)
        if x11: x11.close()
        (output / "result.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

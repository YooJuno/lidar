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

    def enter(x, y, value, ctrl_click=True, commit=True):
        if ctrl_click: key_event("Control_L", True)
        x11.click(window, x, y)
        key_event("Control_L", True)
        key_event("a", True); key_event("a", False)
        key_event("Control_L", False)
        for ch in str(value):
            key = {"-": "minus", ".": "period"}.get(ch, ch)
            key_event(key, True); key_event(key, False)
        if commit:
            key_event("Return", True); key_event("Return", False)
        x11.x.XSync(x11.display, False)
        time.sleep(.2)

    def chord(key):
        key_event("Control_L",True);key_event(key,True);key_event(key,False);key_event("Control_L",False)
        x11.x.XSync(x11.display,False);time.sleep(.25)

    def drag(x1,y1,x2,y2,cancel=False):
        x11.move(window,x1,y1);x11.xt.XTestFakeButtonEvent(x11.display,1,True,0)
        x11.x.XSync(x11.display,False);x11.move(window,x2,y2)
        if cancel: x11.key(window,"Escape")
        x11.xt.XTestFakeButtonEvent(x11.display,1,False,0);x11.x.XSync(x11.display,False);time.sleep(.25)

    def roi_lines(text):
        return [line for line in text.splitlines() if line.startswith("roi_")]

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
            x11.key(window,"r")  # Opens ROI editor without a fragile tab position.
            x11.key(window,"Escape")  # Cancel drawing mode, keeping the editor open.
            x11.click(window,65,290)  # A 4 m box at view center, ready for direct editing.
            count(125,"create_box_at_view_center",1)
            x11.click(window,120,564)  # Exact fields are optional, below the direct controls.
            user_profile=Path(env["XDG_CONFIG_HOME"])/"lidar_viewer/profiles/default.cfg"
            if user_profile.is_file(): profile=user_profile
            enter(270,580,-2,False)
            for x,value in [(75,4),(170,2),(270,2)]: enter(x,641,value,False)
            enter(120,700,90,False)
            count(45,"direct_center_size_and_yaw_input")
            enter(120,700,45,False)
            count(21,"yaw_edit_changes_crop")
            enter(120,700,90,False)
            count(45,"yaw_restore")
            x11.click(window,163,290)  # Distinct, adjacent duplicate.
            count(81,"duplicate_is_offset_and_keeps_dimensions",2)
            enter(75,580,8,False);enter(170,580,0,False);enter(120,700,0,False)
            count(90,"multiple_boxes_union",2)
            capture(x11,window,output/"01_editor.png")
            x11.click(window,201,290)
            count(45,"delete_selected_box",1)
            chord("z");count(90,"undo_delete",2)
            chord("y");count(45,"redo_delete",1)
            chord("z");count(90,"restore_union",2)
            # A click on Delete must not merge into the preceding text edit.
            enter(120,519,"nearby",False,commit=False)
            x11.click(window,201,290)
            chord("z");count(90,"undo_delete_preserves_unsubmitted_name",2)
            result["checks"]["name_edit_and_delete_are_separate_transactions"]='"nearby"' in profile.read_text()
            x11.click(window,46,212);count(3125,"crop_off_restores_full_cloud")
            x11.click(window,46,212);count(90,"crop_on_restores_union")
            # Draw a new footprint, preserving configured raw height and live freeze semantics.
            x11.key(window,"r")
            drag(800,300,1100,570)
            x11.click(window,170,850)
            text=profile.read_text();match=re.search(r"roi_count=(\d+)",text)
            result["checks"]["canvas_drag_creates_third_box"]=bool(match and int(match[1])==3)
            capture(x11,window,output/"02_drawn.png")
            chord("z");count(90,"undo_draw",2)
            chord("y")
            x11.click(window,170,850)
            before_move=profile.read_text()
            # Direct manipulation changes geometry without typing coordinates.
            drag(1100,570,1140,600)
            x11.click(window,170,850)
            resized=profile.read_text()
            result["checks"]["corner_handle_resizes_box"]=roi_lines(resized)!=roi_lines(before_move)
            chord("z");x11.click(window,170,850)
            result["checks"]["undo_resize_restores_exact_box"]=roi_lines(profile.read_text())==roi_lines(before_move)
            drag(950,270,1115,435)
            x11.click(window,170,850)
            last=[line for line in profile.read_text().splitlines() if line.startswith("roi_box=")][-1]
            result["checks"]["round_handle_rotates_box"]=abs(abs(float(last.split()[-1]))-90)<2
            chord("z");x11.click(window,170,850)
            result["checks"]["undo_rotation_restores_exact_box"]=roi_lines(profile.read_text())==roi_lines(before_move)
            # In top view the footprint is at (800,300)..(1100,570).
            x11.click(window,485,102)  # Canvas Move tool.
            drag(950,420,1000,450)
            x11.click(window,170,850)
            result["checks"]["canvas_move_changes_box_center"]=profile.read_text()!=before_move
            chord("z");x11.click(window,170,850)
            result["checks"]["undo_move_restores_exact_profile_roi"]=roi_lines(profile.read_text())==roi_lines(before_move)
            # Esc during a gesture restores the complete previous ROI set.
            x11.key(window,"r");drag(700,250,800,350,cancel=True)
            x11.click(window,170,850)
            result["checks"]["escape_cancels_draw"]=roi_lines(profile.read_text())==roi_lines(before_move)
            # A screen-aligned rectangle remains usable after rotating the top view.
            x11.click(window,1300,700)
            for _ in range(3): x11.key(window,"e")
            x11.key(window,"r");drag(800,300,1100,570)
            x11.click(window,170,850)
            rotated_boxes=[line for line in profile.read_text().splitlines() if line.startswith("roi_box=")]
            result["checks"]["drawing_follows_rotated_top_view"]=len(rotated_boxes)==4 and abs(float(rotated_boxes[-1].split()[-1])-15)<.1
            chord("z");x11.click(window,170,850)
            result["checks"]["undo_rotated_draw_preserves_original_boxes"]=roi_lines(profile.read_text())==roi_lines(before_move)
            # Ground runs once for this frame; subsequent ROI changes reuse it.
            x11.click(window,155,140);x11.click(window,46,222)
            x11.key(window,"r");x11.key(window,"Escape")
            enter(75,580,1,False);x11.click(window,170,850)
            enter(75,580,2,False);x11.click(window,170,850)
            passes=re.findall(r"ROI applied: .*ground passes (\d+)",(output/"viewer.log").read_text())
            result["checks"]["roi_edits_reuse_ground_segmentation"]=len(passes)>=2 and passes[-2:]==["1","1"]
            capture(x11,window,output/"03_cached_ground.png")
            # Check compact and HiDPI variants separately through ui_gl_smoke.
            if not all(result["checks"].values()): raise AssertionError(str(result["checks"]))
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
            if not all(result["checks"].values()): raise AssertionError(str(result["checks"]))
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

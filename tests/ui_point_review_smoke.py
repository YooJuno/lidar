#!/usr/bin/env python3
"""Private Xvfb: independent point/label review while a bag keeps playing."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
import numpy as np
from ui_gl_smoke import X11,capture,changed_pixels,settings_inventory
from ui_controls import Controls
from ui_playback_smoke import write_fixture
from training_adapter_smoke import fixture


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='review-',dir=args.output.resolve()));private=out/'private';(private/'build').mkdir(parents=True)
    shutil.copy2(args.viewer.resolve(),private/'build/lidar_studio');shutil.copytree(repo/'config',private/'config')
    (private/'config/profiles/rosbag.cfg').write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\nroi_count=0\nloop_playback=1\n')
    bag=out/'bag';write_fixture(bag);data=out/'Dataset_with_spaces';fixture(data)
    for n in range(2):
        p=data/f'group{n}/metadata/sample.json';info=json.loads(p.read_text());info['classes']=['background','person','car'];p.write_text(json.dumps(info))
    raw=out/'raw';raw.mkdir();np.array([[1,2,3,50],[2,3,4,90]],dtype='<f4').tofile(raw/'sample.bin');shutil.copy2(raw/'sample.bin',raw/'second.bin')
    env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),LIDAR_VIEWER_UI_TRACE='1',GSETTINGS_BACKEND='memory',NO_AT_BRIDGE='1')
    log_path=out/'parent.log';child_log=out/'settings/lidar_viewer/point_review.log'
    before=settings_inventory(repo);result=dict(status='started',checks={},output=str(out));process=x11=parent=child=None
    def check(name,ok):
        result['checks'][name]=bool(ok)
        if not ok:raise AssertionError(name)
    def wait(condition,timeout=25):
        end=time.monotonic()+timeout
        while not condition():
            if process.poll()is not None:raise RuntimeError(log_path.read_text(errors='replace')[-2000:])
            if time.monotonic()>end:raise TimeoutError('Independent review GUI timed out')
            time.sleep(.06)
    def activate(window):
        x11.x.XRaiseWindow.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
        x11.x.XRaiseWindow(x11.display,window);x11.focus(window);x11.x.XSync(x11.display,False);time.sleep(.12)
    def review_window():return next((w for w in x11.children(x11.root)if x11.title(w)=='Point cloud review - LiDAR Studio'),None)
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with log_path.open('wb')as log:process=subprocess.Popen([str(private/'build/lidar_studio'),str(bag),'--play','--2d'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
        wait(lambda:x11.viewer_window()is not None);parent=x11.viewer_window();activate(parent);x11.resize(parent,1600,960)
        wait(lambda:'Opened bag:'in log_path.read_text());main_controls=Controls(x11,parent,log_path)
        main_controls.click('open_dataset');wait(review_window);child=review_window();activate(child);x11.resize(child,1600,960)
        controls=Controls(x11,child,child_log);controls.rect('dataset_review_path');capture(x11,child,out/'01_open_path.png')
        controls.enter('dataset_review_path',str(data));controls.rect('dataset_current_0')
        check('separate_review_window_opens_folder_path',True);controls.rect('dataset_label_colors')
        check('review_shows_point_labels_and_named_classes',True);capture(x11,child,out/'02_labels_and_boxes.png')
        fresh=len(child_log.read_text());controls.click('dataset_next');controls.rect('dataset_current_1',after=fresh)
        fresh=len(child_log.read_text());controls.click('dataset_previous');controls.rect('dataset_current_0',after=fresh)
        check('folder_previous_and_next_work',True)
        x11.click(child,1300,700);fresh=len(child_log.read_text());x11.key(child,'Right');controls.rect('dataset_current_1',after=fresh)
        check('arrow_keys_work_from_review_canvas',True)
        # Parent remains alive and advances while the child owns keyboard focus.
        activate(parent);a=capture(x11,parent,out/'03_parent_playing.png');time.sleep(.5);b=capture(x11,parent,out/'04_parent_later.png')
        check('parent_bag_keeps_playing_while_review_is_open',changed_pixels(a,b,(400,180,1540,750))>80)
        activate(child);fresh=len(child_log.read_text());controls.enter('dataset_folder_path',str(data/'group0/points/sample.npy'))
        wait(lambda:'Opened point cloud:'in child_log.read_text()[fresh:])
        controls.rect('dataset_current_0',after=fresh);check('same_path_field_accepts_one_point_file',True);controls.rect('dataset_label_colors',after=fresh)
        check('single_file_keeps_sibling_point_labels',True)
        check('single_file_does_not_include_siblings','UI dataset_sample_1 'not in child_log.read_text().rsplit('Opened point cloud:',1)[-1])
        fresh=len(child_log.read_text());controls.enter('dataset_folder_path',str(raw));wait(lambda:'Opened point cloud:'in child_log.read_text()[fresh:]);controls.rect('dataset_current_0',after=fresh)
        check('raw_xyzi_bin_folder_is_viewable',True)
        fresh=len(child_log.read_text());controls.enter('dataset_folder_path',str(raw/'sample.bin'));wait(lambda:'Opened point cloud:'in child_log.read_text()[fresh:])
        controls.rect('dataset_current_0',after=fresh)
        fresh=len(child_log.read_text());controls.click('dataset_folder_path');x11.key(child,'Return');wait(lambda:'Opened point cloud:'in child_log.read_text()[fresh:])
        check('file_path_stays_a_single_file_when_reloaded','UI dataset_sample_1 'not in child_log.read_text().rsplit('Opened point cloud:',1)[-1])
        # ROI shortcuts do not mutate read-only review data.
        x11.click(child,1300,700);x11.key(child,'r');time.sleep(.2);check('review_has_no_roi_or_detection_tabs','UI tab_ROI 'not in child_log.read_text()and 'UI tab_Objects 'not in child_log.read_text())
        x11.key(child,'Escape');wait(lambda:review_window()is None);child=None
        check('closing_review_leaves_main_window_running',process.poll()is None and x11.viewer_window()==parent)
        activate(parent);main_controls.click('open_dataset');wait(review_window);child=review_window();check('review_can_reopen_after_child_exit',True)
        activate(child);Controls(x11,child,child_log).rect('dataset_review_path');x11.key(child,'Escape');time.sleep(.2);x11.click(child,1300,700);x11.key(child,'Escape');wait(lambda:review_window()is None);child=None
        check('original_configuration_unchanged',settings_inventory(repo)==before);result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and child:
            try:capture(x11,child,out/'failure.png')
            except Exception:pass
    finally:
        if x11 and child:
            try:activate(child);x11.key(child,'Escape');x11.key(child,'Escape');time.sleep(.3)
            except Exception:pass
        if process and process.poll()is None:process.terminate();process.wait(timeout=5)
        if x11:x11.close()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed'else 1


if __name__=='__main__':raise SystemExit(main())

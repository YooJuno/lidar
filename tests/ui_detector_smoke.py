#!/usr/bin/env python3
"""Private Xvfb: asynchronous detector protocol integration and playback.
The test worker exercises UI scheduling only; real model correctness is tested
by test_centerpoint.py --model-dir with the original downloaded ONNX weights.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import traceback
from ui_gl_smoke import X11,capture,changed_pixels,settings_inventory
from ui_playback_smoke import write_fixture


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True);out=Path(tempfile.mkdtemp(prefix='detector-ui-',dir=args.output.resolve()))
    private=out/'private';(private/'build').mkdir(parents=True);shutil.copytree(repo/'config',private/'config')
    (private/'python').mkdir()
    (private/'config/profiles/rosbag.cfg').write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=1\nroi_enabled=0\nroi_count=0\nimu_level_enabled=0\ncalibration_z_m=.57\ncolor_mode=0\n')
    # Versioned protocol fixture, deliberately slower than some UI frames.
    (private/'python/detector_worker.py').write_text('''import sys,struct,time,os
from pathlib import Path
with (Path(__file__).parent/'starts.txt').open('a') as log:log.write(str(os.getpid())+'\\n')
source=sys.stdin.buffer;sink=sys.stdout.buffer
q=struct.Struct('<4sQIf');r=struct.Struct('<4sQIfI')
def exact(n):
 b=b''
 while len(b)<n:
  c=source.read(n-len(b))
  if not c:raise SystemExit()
  b+=c
 return b
time.sleep(.4);sink.write(r.pack(b'LDR1',0,0,0,20)+b'CPUExecutionProvider');sink.flush()
while True:
 magic,revision,count,score=q.unpack(exact(q.size));exact(count*12);time.sleep(.07)
 sink.write(r.pack(b'LDR1',revision,2,70,0)+struct.pack('<9f',-3,2,.8,1,.7,1.6,.5,.8,4)+struct.pack('<9f',4,2,.8,4,2,1.6,0,.8,0));sink.flush()
''')
    viewer=private/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
    bag=out/'fixture';write_fixture(bag)
    config=out/'settings/lidar_viewer';config.mkdir(parents=True)
    (config/'detector.cfg').write_text(f'python={os.sys.executable}\nmodel_dir={private}\nprovider=cpu\n')
    env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),LIDAR_VIEWER_DETECTOR_TRACE='1',GSETTINGS_BACKEND='memory',NO_AT_BRIDGE='1')
    before=settings_inventory(repo);result={'status':'started','checks':{},'output':str(out)}
    process=x11=window=None;log_path=out/'viewer.log'
    def rows():return [tuple(map(int,m)) for m in re.findall(r'Detector frame (\d+) revision (\d+) boxes (\d+) full (\d+)',log_path.read_text(errors='replace'))]
    def wait(condition,timeout=20):
        deadline=time.monotonic()+timeout
        while not condition():
            if process.poll() is not None:raise RuntimeError(log_path.read_text(errors='replace'))
            if time.monotonic()>deadline:raise AssertionError('Timed out waiting for detector UI')
            time.sleep(.05)
    def check(name,value):
        result['checks'][name]=bool(value)
        if not value:raise AssertionError(name)
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with log_path.open('wb') as log:
            process=subprocess.Popen([str(viewer),str(bag),'--2d','--play','--detect'],env=env,stdout=log,stderr=subprocess.STDOUT)
            wait(lambda:x11.viewer_window() is not None);window=x11.viewer_window();x11.focus(window);x11.resize(window,1600,960)
            icon=subprocess.check_output(['xprop','-id',str(window),'_NET_WM_ICON'],text=True,env=dict(env,LC_ALL='C'))
            check('native_window_has_multi_resolution_app_icon','32 x 32' in icon and '128 x 128' in icon)
            wait(lambda:len(rows())>=5);initial=rows()[-1]
            for point in [(280,140),(50,140),(155,140),(205,140),(280,140)]:x11.click(window,*point)
            wait(lambda:rows()[-1][0]>initial[0]+3)
            check('tabs_do_not_pause_detected_playback',True)
            capture(x11,window,out/'01_playing_objects.png')
            x11.click(window,410,837);time.sleep(.5);paused=rows();time.sleep(.5)
            check('explicit_pause_holds_frame_and_reuses_result',rows()==paused)
            before_score=capture(x11,window,out/'score_before.png')
            x11.click(window,310,562);time.sleep(.3)
            after_score=capture(x11,window,out/'score_after.png')
            check('score_filter_is_instant_without_reinference',rows()==paused)
            check('higher_score_hides_actual_boxes',changed_pixels(before_score,after_score,(380,180,1540,750))>80)
            x11.click(window,125,562);time.sleep(.2)
            all_targets=capture(x11,window,out/'targets_all.png')
            x11.click(window,108,412);time.sleep(.2)
            no_targets=capture(x11,window,out/'targets_none.png')
            check('clearing_targets_hides_boxes_without_inference',rows()==paused and changed_pixels(all_targets,no_targets,(380,180,1540,750))>80)
            x11.click(window,48,449);time.sleep(.2)
            person_targets=capture(x11,window,out/'targets_person.png')
            check('person_only_filters_cached_model_classes',rows()==paused and changed_pixels(no_targets,person_targets,(380,180,1540,750))>80)
            x11.click(window,150,851);time.sleep(.2)
            saved=private/'config/profiles/rosbag.cfg'
            user=out/'settings/lidar_viewer/profiles/rosbag.cfg'
            if user.is_file():saved=user
            check('selected_classes_are_saved_in_profile','detection_classes=8\n' in saved.read_text())
            # Geometric methods never present their boxes as person/car labels.
            def method(index):
                x11.click(window,175,294);x11.key(window,'Home')
                for _ in range(index):x11.key(window,'Down')
                x11.key(window,'Return');time.sleep(.4)
            method(1);capture(x11,window,out/'clusters_3d.png')
            check('geometric_method_does_not_invoke_neural_worker',rows()==paused)
            method(0);time.sleep(.3)
            check('method_switch_reuses_same_model_process',len((private/'python/starts.txt').read_text().splitlines())==1)
            x11.click(window,48,227);time.sleep(.2);x11.click(window,48,227);time.sleep(.3)
            check('disable_enable_reuses_model_and_current_result',rows()==paused and len((private/'python/starts.txt').read_text().splitlines())==1)
            time.sleep(.2);settled=len(rows());x11.click(window,205,140);x11.click(window,70,328);time.sleep(.4)
            check('roi_edit_reuses_full_scene_neural_result',len(rows())==settled)
            x11.click(window,410,837);wait(lambda:len(rows())>settled+3)
            check('resume_continues_current_revision_detection',True)
            capture(x11,window,out/'02_resumed_roi.png')
            check('original_profiles_unchanged',before==settings_inventory(repo));result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try:capture(x11,window,out/'failure.png')
            except Exception:pass
    finally:
        if process and process.poll() is None:process.terminate();process.wait(timeout=5)
        if x11:x11.close()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1

if __name__=='__main__':raise SystemExit(main())

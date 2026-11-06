#!/usr/bin/env python3
"""Private Xvfb: in-app review, camera preservation, browsing and live bag playback."""
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
import numpy as np
from ui_gl_smoke import X11,capture,changed_pixels,settings_inventory
from ui_controls import Controls
from ui_playback_smoke import write_fixture
from training_adapter_smoke import fixture


def main():
    p=argparse.ArgumentParser();p.add_argument('--viewer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    repo=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='review-',dir=args.output.resolve()));private=out/'private';(private/'build').mkdir(parents=True)
    shutil.copy2(args.viewer.resolve(),private/'build/lidar_studio');shutil.copytree(repo/'config',private/'config')
    (private/'config/profiles/rosbag.cfg').write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\nroi_count=0\nloop_playback=1\n')
    bag=out/'bag';write_fixture(bag);data=out/'Dataset_with_spaces';fixture(data)
    for n in range(2):
        path=data/f'group{n}/metadata/sample.json';info=json.loads(path.read_text());info['classes']=['background','person','car'];path.write_text(json.dumps(info))
    raw=out/'numbered';raw.mkdir()
    for n in [15506,1864,15514,1868,15510]:np.array([[1+n/1000,2,3,50],[2+n/1000,3,4,90]],dtype='<f4').tofile(raw/f'scan_frame_{n}.bin')
    selection=out/'picker_path.txt';selection.write_text(str(data));picker_log=out/'picker.jsonl';chooser=out/'chooser';chooser.mkdir()
    (chooser/'zenity').write_text('#!/usr/bin/python3\nimport os,sys,json\nfrom pathlib import Path\nwith open(os.environ["PICKER_LOG"],"a") as f:f.write(json.dumps(sys.argv)+"\\n")\nprint(Path(os.environ["PICKER_PATH"]).read_text())\n');(chooser/'zenity').chmod(0o755)
    env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),LIDAR_VIEWER_UI_TRACE='1',LIDAR_VIEWER_CAMERA_TRACE='1',GSETTINGS_BACKEND='memory',NO_AT_BRIDGE='1',PICKER_LOG=str(picker_log),PICKER_PATH=str(selection),PATH=str(chooser)+os.pathsep+os.environ['PATH'])
    log_path=out/'viewer.log';before=settings_inventory(repo);result=dict(status='started',checks={},output=str(out));process=x11=window=None
    def check(name,ok):
        result['checks'][name]=bool(ok)
        if not ok:raise AssertionError(name)
    def wait(condition,timeout=20):
        end=time.monotonic()+timeout
        while not condition():
            if process.poll()is not None:raise RuntimeError(log_path.read_text(errors='replace')[-2000:])
            if time.monotonic()>end:raise TimeoutError('Review GUI timed out')
            time.sleep(.06)
    def cameras(after=0):return [tuple(map(float,line.split()))for line in re.findall(r'^Review camera (.+)$',log_path.read_text(errors='replace')[after:],re.M)]
    def camera():
        time.sleep(.15);return cameras()[-1]
    def same_view(a,b):return all(abs(x-y)<1e-5 for x,y in zip(a[1:],b[1:]))
    def windows():return [w for w in x11.children(x11.root)if 'LiDAR Studio'in x11.title(w)]
    def loaded(after):wait(lambda:'Opened point cloud:'in log_path.read_text()[after:]);controls.rect('dataset_current_0',after=after)
    def bag_camera(profile="rosbag"):
        controls.chord('s');time.sleep(.15)
        path=out/f'settings/lidar_viewer/profiles/{profile}.cfg'
        if not path.exists():path=private/f'config/profiles/{profile}.cfg'
        return {k:float(v)for line in path.read_text().splitlines()if '='in line for k,v in[line.split('=',1)]if k in ['camera_distance','camera_target_x','camera_target_y','camera_target_z','camera_yaw_rad','camera_pitch_rad']}
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with log_path.open('wb')as log:process=subprocess.Popen([str(private/'build/lidar_studio'),str(bag),'--play','--2d'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
        wait(lambda:x11.viewer_window()is not None);window=x11.viewer_window();x11.focus(window);x11.resize(window,1600,960)
        wait(lambda:'Opened bag:'in log_path.read_text());controls=Controls(x11,window,log_path);bag_view=bag_camera()
        fresh=len(log_path.read_text());controls.click('open_dataset');controls.rect('dataset_review_path',after=fresh)
        check('review_uses_the_existing_app_window',len(windows())==1 and windows()[0]==window);capture(x11,window,out/'01_open_path.png')
        controls.click('dataset_review_browse');controls.click('dataset_browse_folder');loaded(fresh)
        check('native_browser_can_select_a_folder','--directory'in json.loads(picker_log.read_text().splitlines()[-1]));controls.rect('dataset_label_colors')
        check('folder_review_shows_point_labels',True)
        x,y,r,b=controls.rect('dataset_list');check('sample_list_uses_the_available_panel_height',b-y>300)
        capture(x11,window,out/'02_review_layout.png');check('bag_advances_in_the_background','Background frame 0 'in log_path.read_text()[fresh:])
        controls.drag((1100,400),(1170,435));x11.wheel(window,1100,400,3);view=camera()
        fresh=len(log_path.read_text());controls.click('dataset_next');wait(lambda:cameras(fresh)and cameras(fresh)[-1][0]==1)
        check('3d_camera_and_zoom_stay_fixed_on_next_sample',same_view(view,camera()))
        fresh=len(log_path.read_text());controls.click('dataset_previous');wait(lambda:cameras(fresh)and cameras(fresh)[-1][0]==0)
        check('3d_camera_stays_fixed_on_previous_sample',same_view(view,camera()))
        controls.click('camera_2D');controls.drag((1100,400),(1150,430));x11.key(window,'q');x11.wheel(window,1100,400,2);top=camera()
        fresh=len(log_path.read_text());controls.click('dataset_next');wait(lambda:cameras(fresh)and cameras(fresh)[-1][0]==1)
        check('2d_pan_rotation_and_zoom_stay_fixed',same_view(top,camera()))
        controls.click('dataset_fit');fitted=camera();check('fit_is_an_explicit_action',not same_view(top,fitted)and fitted[0]==1 and fitted[-1]==1)
        fresh=len(log_path.read_text());controls.enter('dataset_folder_path',str(raw));loaded(fresh)
        check('bin_folder_is_viewable',True)
        for i,n in enumerate([1864,1868,15506,15510,15514]):
            fresh=len(log_path.read_text());controls.click(f'dataset_sample_{i}');controls.rect(f'dataset_current_{i}',after=fresh)
            controls.rect('dataset_list')
        check('numbered_samples_are_navigable',True)
        fresh=len(log_path.read_text());controls.enter('dataset_folder_path',str(data/'group0/points/sample.npy'));loaded(fresh);controls.rect('dataset_label_colors',after=fresh)
        check('one_point_file_keeps_its_labels',True)
        check('file_input_only_opens_one_sample','UI dataset_sample_1 'not in log_path.read_text().rsplit('Opened point cloud:',1)[-1])
        # Native file mode is available from the same Browse entry point.
        selection.write_text(str(raw/'scan_frame_1864.bin'));controls.chord('o');controls.rect('dataset_review_path');fresh=len(log_path.read_text())
        controls.click('dataset_review_browse');controls.click('dataset_browse_file');loaded(fresh)
        check('native_browser_can_select_a_file','--directory'not in json.loads(picker_log.read_text().splitlines()[-1]))
        x11.click(window,1300,700);saved=camera();fresh=len(log_path.read_text());controls.click('review_back');controls.rect('open_dataset',after=fresh)
        check('return_to_bag_preserves_its_camera',bag_camera()==bag_view)
        a=capture(x11,window,out/'03_bag_after_return.png');time.sleep(.5);b=capture(x11,window,out/'04_bag_still_playing.png')
        check('return_to_bag_preserves_playback',changed_pixels(a,b,(400,180,1540,750))>80)
        fresh=len(log_path.read_text());controls.click('open_dataset');controls.rect('dataset_current_0',after=fresh)
        check('reopening_review_reuses_the_dataset_and_view',same_view(saved,camera())and len(windows())==1)
        controls.click('dataset_list');fresh=len(log_path.read_text());x11.key(window,'Escape');controls.rect('open_dataset',after=fresh)
        check('escape_returns_to_the_bag',process.poll()is None)
        controls.click('camera_3D');selection.write_text(str(data/'group0/points/sample.npy'));fresh=len(log_path.read_text());controls.chord('o');loaded(fresh)
        check('regular_file_open_initializes_an_upright_dataset_view',bag_camera('dataset')['camera_pitch_rad']>0)
        check('original_configuration_unchanged',settings_inventory(repo)==before);result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and window:
            try:capture(x11,window,out/'failure.png')
            except Exception:pass
    finally:
        if process and process.poll()is None:process.terminate();process.wait(timeout=5)
        if x11:x11.close()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed'else 1


if __name__=='__main__':raise SystemExit(main())

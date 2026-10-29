#!/usr/bin/env python3
"""Private Xvfb: classified render ranges, ROI intersection and saved ground view."""
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
from ui_gl_smoke import X11,capture,changed_pixels,settings_inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='ground-view-',dir=args.output.resolve()))
    private=output/'private';(private/'build').mkdir(parents=True);shutil.copytree(repo/'config',private/'config')
    profile=private/'config/profiles/default.cfg'
    profile.write_text('ground_removal_enabled=0\nground_mode=fast_grid\nground_use_cuda=0\nclustering_enabled=0\ndetection_enabled=0\nroi_enabled=0\nroi_count=0\ngrid_enabled=0\ncolor_mode=0\n')
    fixture=private/'cloud.bin'
    points=[(x*.5,y*.5,0,.1) for x in range(-20,21) for y in range(-20,21)]
    points += [(4+x*.06,4+y*.06,.4+z*.1,.9) for x in range(-5,6) for y in range(-5,6) for z in range(15)]
    fixture.write_bytes(b''.join(struct.pack('ffff',*point) for point in points))
    viewer=private/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
    env=dict(os.environ,XDG_CONFIG_HOME=str(output/'isolated'));before=settings_inventory(repo)
    process=x11=window=None;result={'status':'started','checks':{},'output':str(output)}
    def check(label,value):
        result['checks'][label]=bool(value)
        if not value:raise AssertionError(label)
    def event(key,down):
        code=x11.x.XKeysymToKeycode(x11.display,x11.x.XStringToKeysym(key.encode()));x11.xt.XTestFakeKeyEvent(x11.display,code,down,0)
    def save():
        nonlocal profile
        event('Control_L',True);event('s',True);event('s',False);event('Control_L',False);x11.x.XSync(x11.display,False);time.sleep(.2)
        user=Path(env['XDG_CONFIG_HOME'])/'lidar_viewer/profiles/default.cfg'
        if user.is_file():profile=user
    def inspect():
        completed=subprocess.run([str(viewer),str(fixture),'--inspect'],cwd=private,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
        match=re.search(r'Display: (\w+) \| visible (\d+) \| ground (\d+) \| obstacles (\d+) \| ground passes (\d+)',completed.stdout)
        if completed.returncode or not match:raise AssertionError(completed.stdout)
        return match[1],*[int(match[i]) for i in range(2,6)]
    def select(index):
        x11.click(window,170,294);x11.key(window,'Home')
        for _ in range(index):x11.key(window,'Down')
        x11.key(window,'Return');x11.move(window,1450,700);time.sleep(.25)
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with (output/'viewer.log').open('wb') as log:
            process=subprocess.Popen([str(viewer),str(fixture),'--2d'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
            deadline=time.monotonic()+15;window=None
            while not window:
                if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Viewer did not open')
                window=x11.viewer_window();time.sleep(.1)
            x11.focus(window);x11.resize(window,1600,960);x11.click(window,155,140)
            select(1);save();ground=inspect();ground_image=capture(x11,window,output/'01_ground_only.png')
            check('ground_only_enables_segmentation_and_persists',ground[0]=='ground' and ground[1]==ground[2] and ground[2]>0 and ground[3]>0)
            select(2);save();obstacles=inspect();obstacle_image=capture(x11,window,output/'02_obstacles_only.png')
            check('obstacles_only_persists',obstacles[0]=='obstacles' and obstacles[1]==obstacles[3])
            select(0);save();all_points=inspect();capture(x11,window,output/'03_all_points.png')
            check('views_partition_the_same_classification',all_points[1]==ground[1]+obstacles[1] and all_points[2:4]==ground[2:4]==obstacles[2:4])
            check('actual_render_changes_by_class',changed_pixels(ground_image,obstacle_image,(700,220,1200,650))>200)
            def red_count(picture):
                return sum(r>170 and g<90 and b<90 for r,g,b in picture.crop((750,230,930,400)).convert('RGB').getdata())
            check('ground_pixels_are_hidden_in_obstacles_view',red_count(ground_image)>50 and red_count(obstacle_image)<red_count(ground_image)*.1)
            select(1);x11.click(window,46,222);save();disabled=inspect()
            check('disabling_classification_restores_all',disabled[0]=='all' and disabled[1]==len(points) and disabled[2]==0)
        # Exact count checks use the same scene with a smaller 3D ROI.
        base='\n'.join(line for line in profile.read_text().splitlines() if not line.startswith(('roi_','ground_view=','ground_removal_enabled=')))
        roi='\nroi_enabled=1\nroi_count=1\nroi_box=1 "near object" 4 4 .7 4 4 4 0\nground_removal_enabled=0\n'
        counts=[]
        for mode in ['all','ground','obstacles']:
            profile.write_text(base+roi+f'ground_view={mode}\n');counts.append(inspect())
        check('roi_and_class_filters_intersect',counts[0][1]<len(points) and counts[1][1]>0 and counts[2][1]>0 and counts[0][1]==counts[1][1]+counts[2][1])
        check('saved_restricted_view_overrides_disabled_classifier',counts[1][4]==1 and counts[2][4]==1)
        profile.write_text(base+roi+'ground_view=invalid\n')
        check('invalid_view_falls_back_to_all',inspect()[0]=='all')
        check('original_profiles_unchanged',before==settings_inventory(repo));result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try:capture(x11,window,output/'failure.png')
            except Exception:pass
    finally:
        if process and process.poll() is None:process.terminate();process.wait(timeout=5)
        if x11:x11.close()
        (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

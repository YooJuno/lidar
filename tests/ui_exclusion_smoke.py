#!/usr/bin/env python3
"""Private Xvfb: sensor exclusions reach real segmentation and detector IPC.

The protocol recorder checks the bytes sent to inference, without substituting
any production model. Ground comparisons run each original algorithm against
an independently filtered reference cloud.
"""
import argparse
import collections
import hashlib
import json
import math
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='exclusions-',dir=args.output.resolve()))
    private=out/'private';(private/'build').mkdir(parents=True)
    shutil.copytree(repo/'config',private/'config');(private/'python').mkdir()
    viewer=private/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
    profile=private/'config/profiles/default.cfg'
    user_profile=out/'settings/lidar_viewer/profiles/default.cfg'
    config=out/'settings/lidar_viewer';config.mkdir(parents=True)
    (config/'detector.cfg').write_text(f'python={os.sys.executable}\nmodel_dir={private}\nprovider=cpu\n')
    (private/'python/detector_worker.py').write_text('''import struct,sys,os
from pathlib import Path
root=Path(__file__).parent
with (root/'starts.txt').open('a') as f:f.write(str(os.getpid())+'\\n')
source=sys.stdin.buffer;sink=sys.stdout.buffer
q=struct.Struct('<4sQIf');r=struct.Struct('<4sQIfI')
def exact(n):
 b=b''
 while len(b)<n:
  c=source.read(n-len(b))
  if not c:raise SystemExit()
  b+=c
 return b
sink.write(r.pack(b'LDR1',0,0,0,20)+b'CPUExecutionProvider');sink.flush()
while True:
 tag,revision,count,score=q.unpack(exact(q.size));data=exact(count*12)
 if tag!=b'LDQ1':raise SystemExit(2)
 (root/'last-input.bin').write_bytes(data)
 with (root/'requests.txt').open('a') as f:f.write(str(revision)+' '+str(count)+'\\n')
 sink.write(r.pack(b'LDR1',revision,0,0,0));sink.flush()
''')
    points=[(x*.5,y*.5,-.57,.1) for x in range(-20,21) for y in range(-20,21)]
    points += [(x*.1+.05,y*.1+.05,z*.1+.05,.7) for x in range(-4,5) for y in range(-4,5) for z in range(5)]
    points += [(3+x*.08,2+y*.08,.1+z*.1,.9) for x in range(-3,4) for y in range(-3,4) for z in range(10)]
    fixture=private/'cloud.bin';fixture.write_bytes(b''.join(struct.pack('<4f',*p) for p in points))
    raw_hash=hashlib.sha256(fixture.read_bytes()).hexdigest()
    base=('profile=default\nclustering_enabled=0\ndetection_enabled=1\ndetection_model=tiny\n'
          'calibration_yaw_deg=0\ncalibration_pitch_deg=0\ncalibration_roll_deg=0\ncalibration_z_m=0\n'
          'ground_removal_enabled=1\nground_mode=fast_grid\nground_use_cuda=0\nimu_level_enabled=0\n')
    profile.write_text(base+'roi_enabled=0\nroi_count=0\n')
    env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),LIDAR_VIEWER_ROI_TRACE='1',
             GSETTINGS_BACKEND='memory',NO_AT_BRIDGE='1')
    before=settings_inventory(repo);result={'status':'started','checks':{},'output':str(out)}
    process=x11=window=None;log_path=out/'viewer.log';deadline=time.monotonic()+120

    def check(label,value):
        result['checks'][label]=bool(value)
        if not value:raise AssertionError(label)
    def inspect(cloud=fixture,detect=False):
        completed=subprocess.run([str(viewer),str(cloud),'--inspect']+(['--detect'] if detect else []),
            cwd=private,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
        display=re.search(r'Display: \w+ \| visible (\d+) \| ground (\d+) \| obstacles (\d+) \| ground passes (\d+)',completed.stdout)
        exclusion=re.search(r'Exclusions: removed (\d+) before analysis \| detector input (\d+) points',completed.stdout)
        if completed.returncode or not display or not exclusion:raise AssertionError(completed.stdout)
        return tuple(map(int,display.groups())),tuple(map(int,exclusion.groups()))
    def write_profile(text):
        # Builds with and without USER_SETTINGS read exactly the same fixture.
        profile.write_text(text)
        if user_profile.exists():user_profile.write_text(text)
    def rows():
        return [tuple(map(int,m)) for m in re.findall(r'Analysis input: excluded (\d+) \| detector (\d+) \| ground passes (\d+)',log_path.read_text(errors='replace'))]
    def requests():
        path=private/'python/requests.txt'
        return path.read_text().splitlines() if path.exists() else []
    def wait(predicate):
        while not predicate():
            if process.poll() is not None:raise RuntimeError(log_path.read_text(errors='replace'))
            if time.monotonic()>deadline:raise TimeoutError('Exclusion GUI timed out')
            time.sleep(.05)
    def event(key,down):
        code=x11.x.XKeysymToKeycode(x11.display,x11.x.XStringToKeysym(key.encode()))
        x11.xt.XTestFakeKeyEvent(x11.display,code,down,0)
    def chord(key):
        event('Control_L',True);event(key,True);event(key,False);event('Control_L',False)
        x11.x.XSync(x11.display,False);time.sleep(.25)
    def save():
        chord('s')
        return (user_profile if user_profile.exists() else profile).read_text()
    def changed(action,removed):
        count=len(rows());request_count=len(requests())
        action();wait(lambda:len(rows())>count and rows()[-1][0]==removed)
        wait(lambda:len(requests())>request_count)
        return rows()[-1]
    def drag(a,b):
        x11.move(window,*a);x11.xt.XTestFakeButtonEvent(x11.display,1,True,0);x11.x.XSync(x11.display,False)
        x11.move(window,*b);x11.xt.XTestFakeButtonEvent(x11.display,1,False,0);x11.x.XSync(x11.display,False);time.sleep(.3)

    try:
        # The independent reference uses raw sensor coordinates, not viewer ROI
        # geometry. Rotation/height and automatic BIN floor alignment are active.
        yaw=35*math.pi/180
        def outside(p):
            x,y,z=p[:3];return abs(x*math.cos(yaw)+y*math.sin(yaw))>.5 or abs(-x*math.sin(yaw)+y*math.cos(yaw))>.5 or abs(z)>.5
        filtered=[p for p in points if outside(p)]
        reference=private/'reference.bin';reference.write_bytes(b''.join(struct.pack('<4f',*p) for p in filtered))
        mask='roi_enabled=1\nroi_count=1\nroi_exclude_box=1 "mount" 0 0 0 1 1 1 35\n'
        rotations='calibration_yaw_deg=19\ncalibration_pitch_deg=-12\ncalibration_roll_deg=9\ncalibration_z_m=.57\n'
        for mode in ['fast_grid','gseg3d','patchworkpp','fugseg','grid','ransac']:
            algorithm=base+f'ground_mode={mode}\n'+rotations
            write_profile(algorithm+mask);actual,counts=inspect()
            write_profile(algorithm+'roi_enabled=0\nroi_count=0\n');expected,_=inspect(reference)
            result.setdefault('segmentation_reference',{})[mode]={'actual':actual,'expected':expected,'input_counts':counts}
            check('presegmentation_reference_'+mode,actual==expected and counts==(len(points)-len(filtered),len(filtered)))
        write_profile(base+mask);inspect(detect=True)
        captured=list(struct.iter_unpack('<3f',(private/'python/last-input.bin').read_bytes()))
        expected=collections.Counter(tuple(round(v,5) for v in (p[0],p[1],p[2]+.57)) for p in filtered)
        check('detector_receives_only_unmasked_points',collections.Counter(tuple(round(v,5) for v in p) for p in captured)==expected)
        zeros=private/'zeros.bin';zeros.write_bytes(fixture.read_bytes()+struct.pack('<4f',0,0,0,0))
        write_profile(base+'roi_enabled=0\nroi_count=0\n');_,counts=inspect(zeros,detect=True)
        captured=list(struct.iter_unpack('<3f',(private/'python/last-input.bin').read_bytes()))
        full_expected=collections.Counter(tuple(round(v,5) for v in (p[0],p[1],p[2]+.57)) for p in points)
        check('raw_zero_returns_removed_before_height_alignment',counts==(0,len(points)) and
              collections.Counter(tuple(round(v,5) for v in p) for p in captured)==full_expected)
        # Include regions still crop only the view; the neural input stays whole.
        write_profile(base+'roi_enabled=1\nroi_count=2\nroi_box=1 "far" 3 2 .7 1 1 2 0\n'
                      'roi_exclude_box=1 "mount" 0 0 0 1 1 1 0\n')
        display,counts=inspect()
        check('include_crop_preserves_surrounding_analysis',display[0]==490 and counts==(405,len(points)-405))
        (private/'python/starts.txt').unlink();(private/'python/requests.txt').unlink()
        write_profile(base+'roi_enabled=0\nroi_count=0\n')
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with log_path.open('wb') as log:
            process=subprocess.Popen([str(viewer),str(fixture),'--2d','--detect'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
            wait(lambda:x11.viewer_window() is not None);window=x11.viewer_window();x11.focus(window);x11.resize(window,1600,960)
            wait(lambda:len(requests())==1)
            x11.key(window,'r');x11.key(window,'Escape')
            initial=changed(lambda:x11.click(window,270,328),405)
            check('exclude_button_removes_mount_before_both_analyses',initial[1]==len(points)-405 and initial[2]>=2)
            text=save();check('excluded_box_type_and_sensor_coordinates_saved','roi_exclude_box=1 "Exclude 1" 0 0 0 1 1 1 0' in text)
            capture(x11,window,out/'01_sensor_exclusion.png')
            passes=initial[2]
            changed(lambda:x11.click(window,46,219),0)
            enabled=changed(lambda:x11.click(window,46,219),405)
            check('master_toggle_rebuilds_ground_and_detector',enabled[2]>passes)
            keep=changed(lambda:x11.click(window,270,365),0)
            exclude=changed(lambda:x11.click(window,270,365),405)
            check('keep_exclude_switch_rebuilds_analysis',exclude[2]>keep[2])
            previous=len(requests());passes=exclude[2]
            x11.click(window,295,219);time.sleep(.25)
            check('outline_toggle_reuses_analysis',len(requests())==previous and rows()[-1][2]==passes)
            x11.click(window,295,219)
            previous=len(rows());drag((968,431),(1040,431));wait(lambda:len(rows())>previous)
            moved=rows()[-1];text=save()
            check('canvas_moves_exclusion_and_resegments',moved[2]>passes and 'roi_exclude_box=1' in text and moved[0]!=405)
            changed(lambda:chord('z'),405)
            previous=len(rows());x11.key(window,'r');drag((700,250),(850,350));wait(lambda:len(rows())>previous)
            text=save();check('draw_inherits_exclusion_type',text.count('roi_exclude_box=')==2 and 'roi_count=2' in text)
            changed(lambda:chord('z'),405)
            previous=len(requests());passes=rows()[-1][2]
            x11.click(window,65,328);time.sleep(.35)
            check('adding_keep_region_reuses_ground_and_model',len(requests())==previous and rows()[-1][2]==passes)
            x11.click(window,201,328);time.sleep(.25)
            changed(lambda:x11.click(window,201,328),0)
            text=save();check('deleting_last_exclusion_restores_full_cloud','roi_count=0' in text)
            changed(lambda:chord('z'),405);save()
            check('editing_exclusions_reuses_detector_process',len((private/'python/starts.txt').read_text().splitlines())==1)
            capture(x11,window,out/'02_restored_exclusion.png')
        process.terminate();process.wait(timeout=5);process=None
        _,counts=inspect();check('exclusion_survives_restart',counts==(405,len(points)-405))
        check('raw_file_is_unchanged',hashlib.sha256(fixture.read_bytes()).hexdigest()==raw_hash)
        check('original_profiles_unchanged',before==settings_inventory(repo))
        result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try:capture(x11,window,out/'failure.png')
            except Exception:pass
    finally:
        if process and process.poll() is None:
            process.terminate();process.wait(timeout=5)
        if x11:x11.close()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

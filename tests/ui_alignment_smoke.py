#!/usr/bin/env python3
"""Private Xvfb: manual fallback, effective IMU controls and unambiguous resets."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import traceback
from ui_gl_smoke import X11, capture, changed_pixels, settings_inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bag',type=Path,help='Optional raw VN100 NED / XT32 bag for the axis-preset regression')
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='alignment-',dir=args.output.resolve()))
    before=settings_inventory(repo)
    result={'status':'started','checks':{},'output':str(output)}
    x11=process=window=None
    deadline=time.monotonic()+90
    stage=None

    def wait_for(predicate):
        while not predicate():
            if process.poll() is not None: raise RuntimeError('Viewer exited unexpectedly')
            if time.monotonic()>deadline: raise TimeoutError('Alignment GUI check timed out')
            time.sleep(.1)

    def event(key,down):
        code=x11.x.XKeysymToKeycode(x11.display,x11.x.XStringToKeysym(key.encode()))
        x11.xt.XTestFakeKeyEvent(x11.display,code,down,0)

    def enter(y,value):
        x11.click(window,180,y)
        event('Control_L',True);event('a',True);event('a',False);event('Control_L',False)
        for char in str(value):
            key={'-':'minus','.':'period'}.get(char,char)
            event(key,True);event(key,False)
        event('Return',True);event('Return',False);x11.x.XSync(x11.display,False);time.sleep(.2)

    def values(profile,env):
        x11.click(window,170,850)
        user=Path(env['XDG_CONFIG_HOME'])/'lidar_viewer/profiles'/profile.name
        saved=user if user.is_file() else profile
        return dict(line.split('=',1) for line in saved.read_text().splitlines() if '=' in line)

    def check(label,condition):
        result['checks'][label]=bool(condition)
        if not condition: raise AssertionError(label)

    try:
        x11=X11()
        subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        stages=['no_imu']+(['raw_ned'] if args.bag else [])
        for stage in stages:
            folder=output/stage
            (folder/'build').mkdir(parents=True)
            shutil.copytree(repo/'config',folder/'config')
            profile=folder/'config/profiles'/('rosbag.cfg' if stage=='raw_ned' else 'default.cfg')
            profile.write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\n'
                'roi_enabled=0\nroi_count=0\ncalibration_yaw_deg=12\ncalibration_pitch_deg=-7\n'
                'calibration_roll_deg=5\ncalibration_z_m=0.57\n'
                'imu_mount_confirmed=1\nimu_mount_imu_frame=vn100_link_ned\n'
                'imu_mount_lidar_frame=pandar_xt32\nimu_mount_roll_deg=0.7\n')
            viewer=folder/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
            env=dict(os.environ,XDG_CONFIG_HOME=str(folder/'isolated_config'))
            fixture=folder/'points.bin'
            with fixture.open('wb') as stream:
                for x in range(-15,16):
                    for y in range(-12,13):
                        for z in range(-2,4): stream.write(struct.pack('ffff',x,y,z,.5))
            source=args.bag.resolve() if stage=='raw_ned' else fixture
            log_path=folder/'viewer.log'
            with log_path.open('wb') as log:
                process=subprocess.Popen([str(viewer),str(source),'--2d','--imu-level'],cwd=folder,
                    env=env,stdout=log,stderr=subprocess.STDOUT)
                wait_for(x11.viewer_window);window=x11.viewer_window()
                x11.focus(window);x11.resize(window,1600,960);x11.click(window,163,141)
                if stage=='no_imu':
                    initial=capture(x11,window,folder/'01_fallback.png')
                    enter(464,-4);enter(499,8)
                    data=values(profile,env)
                    check('auto_requested_without_imu_keeps_manual_inputs_editable',
                        data['imu_level_enabled']=='1' and float(data['calibration_pitch_deg'])==-4 and float(data['calibration_roll_deg'])==8)
                    edited=capture(x11,window,folder/'02_manual_edit.png')
                    check('manual_fallback_changes_point_geometry',changed_pixels(initial,edited,(390,180,1540,750))>80)
                else:
                    x11.click(window,1000,878)
                    wait_for(lambda:'IMU leveling active:' in log_path.read_text(errors='replace'))
                    before_fix=capture(x11,window,folder/'01_bad_axes.png')
                    enter(464,-30)
                    data=values(profile,env)
                    check('active_imu_preserves_saved_manual_fallback',float(data['calibration_pitch_deg'])==-7)
                    x11.click(window,170,341)  # Explicit NED -> FLU action in the fixed status area.
                    data=values(profile,env)
                    check('ned_axis_preset_updates_only_mount',float(data['imu_mount_roll_deg'])==180 and
                        float(data['calibration_pitch_deg'])==-7 and abs(float(data['calibration_z_m'])-.57)<1e-6)
                    after_fix=capture(x11,window,folder/'02_axes_fixed.png')
                    check('axis_preset_changes_live_display_transform',changed_pixels(before_fix,after_fix,(390,180,1540,750))>200)
                    x11.click(window,46,222);enter(464,-4)
                    data=values(profile,env)
                    check('disabling_imu_restores_editable_manual_angles',data['imu_level_enabled']=='0' and float(data['calibration_pitch_deg'])==-4)
                x11.click(window,100,577)
                data=values(profile,env)
                check(stage+'_rotation_reset_disables_auto_and_keeps_height',data['imu_level_enabled']=='0' and
                    all(float(data['calibration_'+axis+'_deg'])==0 for axis in ('yaw','pitch','roll')) and
                    abs(float(data['calibration_z_m'])-.57)<1e-6)
                x11.click(window,250,577)
                data=values(profile,env)
                check(stage+'_full_reset_clears_height',float(data['calibration_z_m'])==0)
                if stage=='no_imu':
                    enter(429,372);enter(534,99)
                    data=values(profile,env)
                    check('direct_angles_wrap_and_height_stays_bounded',
                        float(data['calibration_yaw_deg'])==12 and float(data['calibration_z_m'])==5)
                capture(x11,window,folder/'03_reset.png')
            process.terminate();process.wait(timeout=5);process=None;window=None
        check('original_profiles_unchanged',before==settings_inventory(repo))
        result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc(),stage=stage)
        if x11 and window and process and process.poll() is None:
            try:capture(x11,window,output/'failure.png')
            except Exception:pass
    finally:
        if process and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
        if x11:x11.close()
        (output/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

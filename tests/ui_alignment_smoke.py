#!/usr/bin/env python3
"""Private Xvfb: manual fallback, effective IMU controls and unambiguous resets."""
import argparse
import json
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import traceback
from ui_gl_smoke import X11, capture, changed_pixels, settings_inventory
from ui_controls import Controls



def write_imu_fixture(folder,lidar_frame):
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2,PointField,Imu
    writer=rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(folder),storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
    for i,(topic,kind) in enumerate([('/b2/points','PointCloud2'),('/vn100/imu/data_ned','Imu')]):
        writer.create_topic(rosbag2_py.TopicMetadata(id=i,name=topic,type='sensor_msgs/msg/'+kind,serialization_format='cdr'))
    for index in range(100):
        stamp=10_000_000_000+index*10_000_000
        imu=Imu();imu.header.frame_id='vn100_link_ned';imu.header.stamp.sec=10;imu.header.stamp.nanosec=index*10_000_000
        imu.orientation_covariance[0]=-1.;imu.linear_acceleration.x=.1778;imu.linear_acceleration.y=.02555;imu.linear_acceleration.z=-9.69239
        writer.write('/vn100/imu/data_ned',serialize_message(imu),stamp)
        if index%10==0:
            cloud=PointCloud2();cloud.header.frame_id=lidar_frame;cloud.header.stamp=imu.header.stamp
            points=[(x*.2,y*.2,-.57,.5) for x in range(-20,30) for y in range(-15,10)]
            points.extend((x*.1+2,y*.1+3,z*.1,.8) for x in range(8) for y in range(8) for z in range(15))
            cloud.height=1;cloud.width=len(points);cloud.point_step=16;cloud.row_step=cloud.width*16
            cloud.fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(['x','y','z','intensity'])]
            cloud.data=b''.join(struct.pack('<4f',*p) for p in points)
            writer.write('/b2/points',serialize_message(cloud),stamp)
    writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bag',type=Path,help='Optional raw VN100 NED / forward XT32 bag for the automatic-axes regression')
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
        c.chord('s')
        user=Path(env['XDG_CONFIG_HOME'])/'lidar_viewer/profiles'/profile.name
        saved=user if user.is_file() else profile
        return dict(line.split('=',1) for line in saved.read_text().splitlines() if '=' in line)

    def check(label,condition):
        result['checks'][label]=bool(condition)
        if not condition: raise AssertionError(label)

    try:
        x11=X11()
        subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        stages=['no_imu','raw_ned_native','raw_ned_forward']+(['raw_ned_real'] if args.bag else [])
        for stage in stages:
            raw=stage.startswith('raw_ned');lidar_frame='pandar_xt32' if stage=='raw_ned_native' else 'pandar_xt32_forward'
            folder=output/stage
            (folder/'build').mkdir(parents=True)
            shutil.copytree(repo/'config',folder/'config')
            profile=folder/'config/profiles'/('rosbag.cfg' if raw else 'default.cfg')
            profile.write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\n'
                'roi_enabled=0\nroi_count=0\ncalibration_yaw_deg=12\ncalibration_pitch_deg=-7\n'
                'calibration_roll_deg=5\ncalibration_z_m=0.57\n'
                'imu_mount_confirmed=1\nimu_mount_imu_frame=vn100_link_ned\n'
                f'imu_mount_lidar_frame={lidar_frame}\nimu_mount_roll_deg=0.7\nimu_mount_yaw_deg=90\nimu_mount_auto_axes=0\n')
            viewer=folder/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
            env=dict(os.environ,XDG_CONFIG_HOME=str(folder/'isolated_config'),LIDAR_VIEWER_UI_TRACE='1')
            fixture=folder/'points.bin'
            with fixture.open('wb') as stream:
                for x in range(-15,16):
                    for y in range(-12,13):
                        for z in range(-2,4): stream.write(struct.pack('ffff',x,y,z,.5))
            source=fixture
            if raw:
                source=args.bag.resolve() if stage=='raw_ned_real' else folder/'imu_fixture'
                if stage!='raw_ned_real':write_imu_fixture(source,lidar_frame)
            log_path=folder/'viewer.log'
            with log_path.open('wb') as log:
                process=subprocess.Popen([str(viewer),str(source),'--2d','--imu-level'],cwd=folder,
                    env=env,stdout=log,stderr=subprocess.STDOUT)
                wait_for(x11.viewer_window);window=x11.viewer_window()
                x11.focus(window);x11.resize(window,1600,960)
                c=Controls(x11,window,log_path)
                if raw:wait_for(lambda: "Opened bag:" in log_path.read_text(errors="replace"))
                x11.click(window,100,141)
                if stage=='no_imu':
                    initial=capture(x11,window,folder/'01_fallback.png')
                    enter(464,-4);enter(499,8)
                    data=values(profile,env)
                    check('auto_requested_without_imu_keeps_manual_inputs_editable',
                        data['imu_level_enabled']=='1' and float(data['calibration_pitch_deg'])==-4 and float(data['calibration_roll_deg'])==8)
                    edited=capture(x11,window,folder/'02_manual_edit.png')
                    check('manual_fallback_changes_point_geometry',changed_pixels(initial,edited,(390,180,1540,750))>80)
                else:
                    wait_for(lambda:'IMU leveling active:' in log_path.read_text(errors='replace'))
                    before_fix=capture(x11,window,folder/'01_bad_axes.png')
                    enter(464,-30)
                    data=values(profile,env)
                    check(stage+'_active_imu_preserves_manual_fallback',float(data['calibration_pitch_deg'])==-7)
                    c.click('align_mounting');x11.wheel(window,200,700,-5);c.click('align_axes')
                    data=values(profile,env)
                    check(stage+'_axis_conversion_preserves_mount_and_height',data['imu_mount_auto_axes']=='1' and
                        abs(float(data['imu_mount_roll_deg'])-.7)<1e-6 and float(data['imu_mount_yaw_deg'])==90 and
                        float(data['calibration_pitch_deg'])==-7 and abs(float(data['calibration_z_m'])-.57)<1e-6)
                    inspected=subprocess.run([str(viewer),str(source),'--inspect','--imu-level'],cwd=folder,env=env,
                        stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=40)
                    angles=re.search(r'IMU correction: roll ([-.\d]+) pitch ([-.\d]+) \| automatic axes (\d)',inspected.stdout)
                    check(stage+'_normalizes_up_without_180_degree_flip',inspected.returncode==0 and angles is not None and
                        abs(float(angles[1]))<5 and abs(float(angles[2]))<5 and angles[3]=='1')
                    x11.wheel(window,200,500,12)
                    after_fix=capture(x11,window,folder/'02_axes_fixed.png')
                    check(stage+'_axes_apply_to_running_display',changed_pixels(before_fix,after_fix,(390,180,1540,750))>200)
                    c.click('align_auto');c.enter('##Pitch',-4)
                    data=values(profile,env)
                    check(stage+'_disabling_imu_restores_manual_input',data['imu_level_enabled']=='0' and float(data['calibration_pitch_deg'])==-4)
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

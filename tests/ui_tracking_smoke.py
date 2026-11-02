#!/usr/bin/env python3
"""Private Xvfb: sensor-time IDs, occlusion, seek/reset and cached UI actions.

The protocol fixture supplies known measurements to test integration with the
native tracker. It does not replace or evaluate the production neural models.
"""
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
from ui_controls import Controls


def write_fixture(folder):
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    writer=rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(folder),storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
    writer.create_topic(rosbag2_py.TopicMetadata(id=0,name='/b2/points',type='sensor_msgs/msg/PointCloud2',serialization_format='cdr'))
    for index in range(30):
        cloud=PointCloud2();cloud.header.frame_id='pandar_xt32_forward'
        cloud.header.stamp.sec=50+index//10;cloud.header.stamp.nanosec=index%10*100_000_000
        points=[(float(index),0.,0.,0.)]
        for cx in [-3+index*.1,3-index*.1]:
            points.extend((cx+x*.1,2+y*.1,z*.1,1.) for x in range(-3,4) for y in range(-2,3) for z in range(18))
        cloud.height=1;cloud.width=len(points);cloud.point_step=16;cloud.row_step=cloud.width*16
        cloud.fields=[PointField(name=n,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,n in enumerate(['x','y','z','intensity'])]
        cloud.data=b''.join(struct.pack('<4f',*p) for p in points)
        writer.write('/b2/points',serialize_message(cloud),50_000_000_000+index*100_000_000)
    writer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--viewer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    repo=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='tracking-',dir=args.output.resolve()))
    private=out/'private';(private/'build').mkdir(parents=True);shutil.copytree(repo/'config',private/'config');(private/'python').mkdir()
    profile=private/'config/profiles/rosbag.cfg'
    profile.write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=1\ndetection_model=tiny\ndetection_min_score=.5\nroi_count=0\nimu_level_enabled=0\ncalibration_z_m=0\ntracking_enabled=1\n')
    (private/'python/detector_worker.py').write_text('''import os,struct,sys,time
from pathlib import Path
root=Path(__file__).parent
with (root/'starts.txt').open('a') as f:f.write(str(os.getpid())+'\\n')
q=struct.Struct('<4sQIf');r=struct.Struct('<4sQIfI');source=sys.stdin.buffer;sink=sys.stdout.buffer
def exact(n):
 b=b''
 while len(b)<n:
  c=source.read(n-len(b))
  if not c:raise SystemExit()
  b+=c
 return b
sink.write(r.pack(b'LDR1',0,0,0,20)+b'CPUExecutionProvider');sink.flush()
while True:
 tag,rev,count,score=q.unpack(exact(q.size));data=exact(count*12);index=round(struct.unpack_from('<f',data)[0])
 boxes=[[-3+index*.1,2,.9,.7,.6,1.8,0,.9,4],[3-index*.1,2,.9,.7,.6,1.8,0,.9,4]]
 if index in [6,7]:boxes.pop(0)
 if index%2:boxes.reverse()
 time.sleep(.02);sink.write(r.pack(b'LDR1',rev,len(boxes),20,0)+b''.join(struct.pack('<9f',*b) for b in boxes));sink.flush()
''')
    viewer=private/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer);bag=out/'fixture';write_fixture(bag)
    config=out/'settings/lidar_viewer';config.mkdir(parents=True)
    (config/'detector.cfg').write_text(f'python={os.sys.executable}\nmodel_dir={private}\nprovider=cpu\n')
    env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),LIDAR_VIEWER_TRACKING_TRACE='1',LIDAR_VIEWER_DETECTOR_TRACE='1',LIDAR_VIEWER_UI_TRACE='1',GSETTINGS_BACKEND='memory',NO_AT_BRIDGE='1')
    log_path=out/'viewer.log';before=settings_inventory(repo);result={'status':'started','checks':{},'output':str(out)};process=x11=window=None
    def log():return log_path.read_text(errors='replace')
    def tracks():
        return [tuple(map(float,m)) for m in re.findall(r'Track frame (\d+) id (\d+) class (\d+) confirmed (\d+) predicted (\d+) speed ([-.\de+]+) x ([-.\de+]+)',log())]
    def frames():return re.findall(r'Detector frame (\d+) revision (\d+)',log())
    def wait(condition,timeout=20):
        deadline=time.monotonic()+timeout
        while not condition():
            if process.poll() is not None:raise RuntimeError(log())
            if time.monotonic()>deadline:raise AssertionError('Timed out waiting for tracker GUI')
            time.sleep(.03)
    def check(name,value):
        result['checks'][name]=bool(value)
        if not value:raise AssertionError(name)
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with log_path.open('wb') as stream:
            process=subprocess.Popen([str(viewer),str(bag),'--2d','--detect'],cwd=private,env=env,stdout=stream,stderr=subprocess.STDOUT)
            wait(lambda:x11.viewer_window() is not None);window=x11.viewer_window();x11.focus(window);x11.resize(window,1600,960);c=Controls(x11,window,log_path)
            wait(lambda:len(tracks())==2);initial=tracks();ids={t[1] for t in initial};left=min(initial,key=lambda t:t[-1])[1]
            for index in range(1,13):
                x11.key(window,'Right');wait(lambda:any(t[0]==index for t in tracks()))
                current=[t for t in tracks() if t[0]==index]
                check(f'frame_{index}_keeps_ids_despite_order_or_occlusion',{t[1] for t in current}==ids)
                if index in [6,7]:check(f'frame_{index}_predicts_missing_object',any(t[1]==left and t[4]==1 for t in current))
            last=[t for t in tracks() if t[0]==12]
            check('reacquisition_confirms_same_ids_and_sensor_time_speed',all(t[3]==1 and t[4]==0 and abs(t[5]-1)<.2 for t in last))
            paused=tracks();requests=frames();time.sleep(1.1)
            check('pause_does_not_age_tracks',tracks()==paused and frames()==requests)
            x11.click(window,280,140);x11.click(window,310,562);time.sleep(.2);x11.click(window,125,562);x11.click(window,108,412);time.sleep(.2)
            check('score_and_class_filters_do_not_age_or_rerun',tracks()==paused and frames()==requests)
            x11.click(window,48,412)
            x11.key(window,'r');c.drag((800,300),(1100,570));c.click('roi_apply');time.sleep(.4)
            check('roi_apply_reuses_neural_result_and_tracking',tracks()==paused and frames()==requests)
            x11.click(window,280,140);capture(x11,window,out/'01_confirmed_tracks.png')
            c.click('tracking_ids');c.click('tracking_ids');wait(lambda:len(tracks())>len(paused))
            check('ids_toggle_keeps_worker_and_inference',frames()==requests and len((private/'python/starts.txt').read_text().splitlines())==1)
            check('explicit_reenable_allocates_fresh_ids',not ({t[1] for t in tracks()[-2:]} & ids))
            old_ids={t[1] for t in tracks()[-2:]};x11.key(window,'Left');wait(lambda:len(frames())>len(requests))
            check('backward_seek_resets_ids',not ({t[1] for t in tracks()[-2:]} & old_ids) and 'Track reset: seek or loop' in log())
            c.chord('s');saved=out/'settings/lidar_viewer/profiles/rosbag.cfg'
            check('tracking_preference_is_saved','tracking_enabled=1\n' in (saved if saved.exists() else profile).read_text())
            # Real playback advances detections while sidebar controls own focus.
            old=len(frames());x11.key(window,'space');wait(lambda:len(frames())>=old+5);x11.key(window,'space');time.sleep(.3)
            check('tracking_playback_keeps_one_inference_process',len((private/'python/starts.txt').read_text().splitlines())==1)
            samples=[float(s) for s in re.findall(r'^Status sample ([.\de+]+)$',log(),re.M)]
            check('status_metrics_refresh_at_readable_intervals',len(samples)>=4 and all(b-a>=.99 for a,b in zip(samples,samples[1:])))
            capture(x11,window,out/'02_sidebar_playback.png')
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

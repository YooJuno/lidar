#!/usr/bin/env python3
"""Private Xvfb + synthetic bag: controls stay responsive and ROI drags resume playback."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import traceback
from ui_gl_smoke import X11, capture, changed_pixels, settings_inventory


def write_fixture(folder):
    import rosbag2_py
    from rclpy.serialization import serialize_message
    from sensor_msgs.msg import PointCloud2, PointField
    writer=rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(folder),storage_id='mcap'),rosbag2_py.ConverterOptions('cdr','cdr'))
    writer.create_topic(rosbag2_py.TopicMetadata(id=0,name='/b2/points',type='sensor_msgs/msg/PointCloud2',serialization_format='cdr'))
    for index in range(120):
        cloud=PointCloud2();cloud.header.frame_id='pandar_xt32'
        shift=4*math.sin(index*.17)
        points=[(x*.3,y*.3,-.57,20.) for x in range(-20,21) for y in range(-20,21)]
        points += [(shift+x*.08,2+y*.08,z*.08,80.) for x in range(-5,6) for y in range(-5,6) for z in range(8)]
        cloud.height=1;cloud.width=len(points);cloud.point_step=16;cloud.row_step=16*cloud.width
        cloud.fields=[PointField(name=name,offset=i*4,datatype=PointField.FLOAT32,count=1) for i,name in enumerate(['x','y','z','intensity'])]
        cloud.data=b''.join(struct.pack('<ffff',*p) for p in points)
        writer.write('/b2/points',serialize_message(cloud),1_000_000_000+index*100_000_000)
    writer.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='playback-',dir=args.output.resolve()))
    private=output/'private';(private/'build').mkdir(parents=True)
    shutil.copytree(repo/'config',private/'config')
    (private/'config/profiles/rosbag.cfg').write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\nroi_enabled=0\nroi_count=0\nimu_level_enabled=0\ncolor_mode=0\n')
    viewer=private/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
    bag=output/'fixture';write_fixture(bag)
    env=dict(os.environ,XDG_CONFIG_HOME=str(output/'isolated'),GSETTINGS_BACKEND='memory',NO_AT_BRIDGE='1')
    before=settings_inventory(repo);result={'status':'started','checks':{},'output':str(output)}
    process=x11=window=None;deadline=time.monotonic()+65;log_path=output/'viewer.log'
    crop=(390,180,1540,750);header=(1235,20,1320,48)
    def wait_for(predicate):
        while not predicate():
            if process.poll() is not None: raise RuntimeError('Viewer exited unexpectedly')
            if time.monotonic()>deadline: raise TimeoutError('Playback UI timed out')
            time.sleep(.05)
    def check(label,value):
        result['checks'][label]=bool(value)
        if not value: raise AssertionError(label)
    def shot(name): return capture(x11,window,output/(name+'.png'))
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with log_path.open('wb') as log:
            process=subprocess.Popen([str(viewer),str(bag),'--play','--2d','--sensor-height','.57'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
            wait_for(x11.viewer_window);window=x11.viewer_window();x11.focus(window);x11.resize(window,1600,960)
            wait_for(lambda:'Opened bag:' in log_path.read_text(errors='replace'));time.sleep(.3)
            playing=shot('01_playing');time.sleep(.35);moving=shot('02_motion')
            check('initial_bag_autoplay_moves',changed_pixels(playing,moving,crop)>80)
            # Ordinary controls, including holding a tab button, keep playback active.
            for x in [50,100,155,255]: x11.click(window,x,140)
            x11.click(window,485,102)
            x11.click(window,65,290);x11.click(window,46,219)  # New box, then crop off.
            x11.click(window,163,290);x11.click(window,201,290)
            x11.click(window,1567,30);x11.key(window,'Escape')
            controls=shot('03_controls');time.sleep(.35);controls_later=shot('04_controls_motion')
            check('controls_and_roi_actions_keep_playing_state',changed_pixels(playing,controls,header)==0)
            check('controls_do_not_freeze_cloud',changed_pixels(controls,controls_later,crop)>80)
            x11.move(window,50,140);x11.xt.XTestFakeButtonEvent(x11.display,1,True,0);x11.x.XSync(x11.display,False)
            held=shot('05_held');time.sleep(.35);held_later=shot('06_held_motion')
            x11.xt.XTestFakeButtonEvent(x11.display,1,False,0);x11.x.XSync(x11.display,False)
            check('holding_ui_button_keeps_frames_advancing',changed_pixels(held,held_later,crop)>80)
            x11.move(window,900,450);x11.key(window,'r')
            check('selecting_draw_tool_keeps_playing',changed_pixels(playing,shot('07_draw_selected'),header)==0)
            x11.move(window,650,250);x11.xt.XTestFakeButtonEvent(x11.display,1,True,0);x11.x.XSync(x11.display,False)
            x11.move(window,1300,650);time.sleep(.2)
            dragging=shot('08_dragging');time.sleep(.35);dragging_later=shot('09_drag_still')
            check('active_roi_drag_holds_geometry',changed_pixels(dragging,dragging_later,crop)==0)
            x11.xt.XTestFakeButtonEvent(x11.display,1,False,0);x11.x.XSync(x11.display,False);time.sleep(.3)
            resumed=shot('10_automatic_resume');time.sleep(.35);resumed_later=shot('11_resumed_motion')
            check('roi_release_restores_playback_state',changed_pixels(playing,resumed,header)==0)
            check('roi_release_restores_motion',changed_pixels(resumed,resumed_later,crop)>80)
            x11.click(window,410,837);frozen=shot('12_explicit_pause');time.sleep(.35)
            check('explicit_pause_still_freezes_cloud',changed_pixels(frozen,shot('13_still_paused'),crop)==0)
            check('original_profiles_unchanged',before==settings_inventory(repo));result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try:shot('failure')
            except Exception:pass
    finally:
        if process and process.poll() is None:
            process.terminate();process.wait(timeout=5)
        if x11:x11.close()
        (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

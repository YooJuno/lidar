#!/usr/bin/env python3
"""Private Xvfb: original-intensity ROI datasets, batch export and IMU metadata."""
import argparse
import hashlib
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
import numpy as np
from ui_gl_smoke import X11, capture, settings_inventory
from ui_controls import Controls
from ui_alignment_smoke import write_imu_fixture


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();repo=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='export-',dir=args.output.resolve()))
    private=out/'private';(private/'build').mkdir(parents=True);shutil.copytree(repo/'config',private/'config')
    viewer=private/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
    exports=out/"exports 'literal' $(false) 한글";exports.mkdir()
    chooser=private/'native';chooser.mkdir()
    (chooser/'zenity').write_text('#!'+os.sys.executable+'\nimport os,sys,json\nfrom pathlib import Path\n'
        "Path(os.environ['CHOOSER_ARGS']).write_text(json.dumps(sys.argv))\n"
        "print(os.environ['EXPORT_FOLDER'])\n")
    (chooser/'zenity').chmod(0o755)
    frames=private/'frames';frames.mkdir()
    for index in range(8):
        points=[(x*.25,y*.25,-.57,32.) for x in range(-20,21) for y in range(-20,21)]
        points += [(x*.12+index*.02,y*.12,.2+z*.12,128.+index) for x in range(-2,3) for y in range(-2,3) for z in range(5)]
        points += [(3+x*.12,2+y*.12,.2+z*.12,512.+index) for x in range(-2,3) for y in range(-2,3) for z in range(5)]
        points += [(0,0,0,0)]
        (frames/f'{index:05d}.bin').write_bytes(b''.join(struct.pack('<4f',*v) for v in points))
    # Hard links give cancellation a real, sufficiently long batch without
    # duplicating point buffers/files or introducing delays into production.
    for index in range(8,5000):os.link(frames/f'{index%8:05d}.bin',frames/f'{index:05d}.bin')
    original_hash=hashlib.sha256((frames/'00000.bin').read_bytes()).hexdigest()
    profile=private/'config/profiles/default.cfg'
    profile.write_text('ground_removal_enabled=1\nground_mode=fast_grid\nground_use_cuda=0\n'
        'clustering_enabled=0\ndetection_enabled=0\nimu_level_enabled=0\n'
        'calibration_yaw_deg=23\ncalibration_pitch_deg=-7\ncalibration_roll_deg=5\ncalibration_z_m=.57\n'
        'roi_count=1\nroi_exclude_box=1 "mount" 0 0 0 1 1 1 0\n'
        'display_roi_enabled=1\ndisplay_roi_count=2\n'
        'display_roi_box=1 "near" 0 0 0 4 4 4 30\n'
        'display_roi_box=1 "far" 3 2 0 2 2 4 -20\n')
    env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),LIDAR_VIEWER_UI_TRACE='1',
        LIDAR_VIEWER_ROI_TRACE='1',EXPORT_FOLDER=str(exports),CHOOSER_ARGS=str(out/'chooser.json'),
        PATH=str(chooser)+os.pathsep+os.environ['PATH'],NO_AT_BRIDGE='1',GSETTINGS_BACKEND='memory')
    result={'status':'started','checks':{},'output':str(out)}
    before=settings_inventory(repo);process=x11=window=c=None;log_path=out/'viewer.log'
    deadline=time.monotonic()+240

    def check(name,value):
        result['checks'][name]=bool(value)
        if not value:raise AssertionError(name)
    def wait(predicate):
        while not predicate():
            if process.poll() is not None:raise RuntimeError(log_path.read_text(errors='replace')[-3000:])
            if time.monotonic()>deadline:raise TimeoutError('Export GUI timed out')
            time.sleep(.03)
    def text():return log_path.read_text(errors='replace')
    def launch(source):
        nonlocal process,window,c
        offset=len(text()) if log_path.exists() else 0
        log=log_path.open('ab')
        process=subprocess.Popen([str(viewer),str(source),'--2d'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
        log.close();window=None;wait(lambda:x11.viewer_window() is not None)
        window=x11.viewer_window();x11.focus(window);x11.resize(window,1600,960)
        wait(lambda:('Opened bag:' if source!=frames else 'Input:') in text()[offset:])
        c=Controls(x11,window,log_path);time.sleep(.3);c.click('tab_ROI')
    def start_export():
        old=set(exports.iterdir());offset=len(text());c.click('export_start')
        wait(lambda:'ROI export: Saved' in text()[offset:] or 'ROI export failed:' in text()[offset:])
        new=set(exports.iterdir())-old
        if len(new)!=1:raise AssertionError('Expected one new dataset; log: '+text()[-2000:])
        path=new.pop();manifest=json.loads((path/'metadata.json').read_text())
        check('export_completed',manifest['status']=='complete')
        return path,manifest
    def expected(source):
        points=np.fromfile(source,dtype='<f4').reshape(-1,4)
        keep=np.zeros(len(points),dtype=bool)
        for center,size,yaw in [([0,0,0],[4,4,4],30),([3,2,0],[2,2,4],-20)]:
            d=points[:,:3]-center;angle=np.deg2rad(yaw);x=d[:,0]*np.cos(angle)+d[:,1]*np.sin(angle)
            y=-d[:,0]*np.sin(angle)+d[:,1]*np.cos(angle)
            keep|=(abs(x)<=size[0]*.5+1e-5)&(abs(y)<=size[1]*.5+1e-5)&(abs(d[:,2])<=size[2]*.5+1e-5)
        keep&=np.any(points[:,:3]!=0,axis=1)
        return points[keep]
    def validate_sample(path,sample,raw_expected,aligned=False):
        points=np.load(path/sample['points'],allow_pickle=False)
        meta=json.loads((path/sample['metadata']).read_text())
        stem=Path(sample['points']).stem
        source_name=Path(meta['source']['path']).stem
        check('sample_names_identify_source_frame_and_roi',stem.startswith(source_name+'_') and
            f'_f{meta["source"]["frame_index"]+1}_' in stem and
            all(region['name'].replace(' ','-') in stem for region in meta['roi_sensor_frame']) and
            meta['source']['frame_number']==meta['source']['frame_index']+1)
        check('paired_formats_keep_matching_names',stem==sample['id']==Path(sample['pcd']).stem==Path(sample['metadata']).stem)
        check('dataset_folder_identifies_source_and_frames',path.name.startswith(source_name+'_f') or '_n' in path.name)
        check('npy_float32_xyzi',points.dtype==np.float32 and points.shape==(len(raw_expected),4))
        binary=(path/sample['pcd']).read_bytes();header,payload=binary.split(b'DATA binary\n',1)
        check('pcd_fields_and_payload_match_numpy',b'FIELDS x y z intensity\n' in header and
            b'TYPE F F F F\n' in header and np.array_equal(np.frombuffer(payload,dtype='<f4').reshape(-1,4),points))
        check('original_intensity_is_preserved',np.array_equal(points[:,3],raw_expected[:,3]) and np.max(points[:,3])>1)
        transform=np.array(meta['sensor_to_output_row_major']).reshape(4,4)
        target=np.c_[raw_expected[:,:3],np.ones(len(raw_expected))]@transform.T
        check('coordinates_match_saved_transform',np.allclose(points[:,:3],target[:,:3],atol=3e-6))
        if not aligned:check('sensor_coordinates_are_original',np.array_equal(points,raw_expected))
        check('export_is_unlabeled_and_independent_of_ground_display',
            meta['annotation_status']=='unlabeled' and not meta['ground_filter_applied'] and not meta['ground_exclusions_applied'])
        return meta
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        launch(frames);c.click('roi_export');c.click('export_browse')
        wait(lambda:(out/'chooser.json').exists());time.sleep(.2)
        check('native_directory_picker_used','--directory' in json.loads((out/'chooser.json').read_text()))
        path,manifest=start_export();sample=manifest['samples'][0]
        check('export_dialog_stays_inside_window',all(0<=v[0]<v[2]<=1600 and 0<=v[1]<v[3]<=960
            for v in [c.rect('export_browse'),c.rect('export_close')]))
        source=frames/f'{sample["source_frame_index"]:05d}.bin'
        validate_sample(path,sample,expected(source))
        check('metadata_preserves_both_rois_and_no_fake_labels',len(manifest['roi_sensor_frame'])==2 and not (path/'labels').exists())
        hints=manifest['labeling']
        check('roi_names_are_default_class_hints',hints['point_annotations']=='not_provided'and
            [item['class_name']for item in hints['regions']]==[item['name']for item in manifest['roi_sensor_frame']])
        capture(x11,window,out/'01_sensor_export.png')
        c.click('export_close')
        direct=out/'quick-exports';direct.mkdir();c.enter('quick_export_folder',str(direct))
        prefs=out/'settings/lidar_viewer/roi_export_folder.txt'
        check('typed_export_folder_is_remembered',prefs.read_text()==str(direct))
        offset=len(text());c.chord('e');wait(lambda:'ROI export: Saved'in text()[offset:])
        quick=list(direct.iterdir());check('shortcut_exports_one_current_frame_without_dialog',len(quick)==1 and
            json.loads((quick[0]/'metadata.json').read_text())['frames_requested']==1)
        c.click('quick_export_browse');wait(lambda:prefs.read_text()==str(exports));time.sleep(.2)
        c.click('roi_export')
        c.enter('export_class_name','person');c.click('export_aligned');path,manifest=start_export();sample=manifest['samples'][0]
        meta=validate_sample(path,sample,expected(frames/f'{sample["source_frame_index"]:05d}.bin'),True)
        check('common_class_overrides_hints_and_keeps_roi_names',meta['category']=='person'and
            all(item['class_name']=='person'and item['source']=='common_class'for item in meta['labeling']['regions'])and
            [item['roi_name']for item in meta['labeling']['regions']]==[item['name']for item in meta['roi_sensor_frame']]and
            meta['labeling']==manifest['labeling']and meta['annotation_status']=='unlabeled')
        check('manual_alignment_and_ground_offset_saved',meta['effective_alignment_deg']=={'yaw':23,'pitch':-7,'roll':5}
            and abs(meta['ground_offset_m']+.57)<1e-6 and abs(meta['height_offset_m']-.57)<1e-6)
        c.click('export_close');x11.key(window,'space');time.sleep(.2)
        c.click('roi_export');c.click('export_range')
        c.enter('export_first',2);c.enter('export_last',6);c.enter('export_stride',2)
        old_passes=len(re.findall('Analysis input:',text()));path,manifest=start_export();time.sleep(.4)
        check('batch_sampling_uses_requested_source_frames',[s['source_frame_index'] for s in manifest['samples']]==[1,3,5])
        for sample in manifest['samples']:validate_sample(path,sample,expected(frames/f'{sample["source_frame_index"]:05d}.bin'),True)
        check('export_preserves_playing_state',len(re.findall('Analysis input:',text()))>old_passes+1)
        capture(x11,window,out/'02_batch_export.png')
        c.enter('export_first',1);c.enter('export_last',5000);c.enter('export_stride',1)
        old=set(exports.iterdir());offset=len(text());c.click('export_start');c.click('export_cancel')
        wait(lambda:'ROI export: Export canceled' in text()[offset:])
        check('cancel_removes_partial_dataset',set(exports.iterdir())==old)
        c.click('export_close');x11.key(window,'space');process.terminate();process.wait(timeout=5);process=None
        # Reuse an actual ROS 2/IMU fixture to verify per-frame leveling,
        # original bag intensity and source timestamp/topic provenance.
        bag=out/'bag';write_imu_fixture(bag,'pandar_xt32_forward')
        (private/'config/profiles/rosbag.cfg').write_text(
            'ground_removal_enabled=1\nground_mode=fast_grid\nground_use_cuda=0\nclustering_enabled=0\ndetection_enabled=0\n'
            'imu_level_enabled=1\nimu_mount_confirmed=1\nimu_mount_auto_axes=1\n'
            'imu_mount_imu_frame=vn100_link_ned\nimu_mount_lidar_frame=pandar_xt32_forward\n'
            'imu_mount_roll_deg=0\nimu_mount_pitch_deg=0\nimu_mount_yaw_deg=0\n'
            'calibration_yaw_deg=0\ncalibration_pitch_deg=0\ncalibration_roll_deg=0\ncalibration_z_m=.57\n'
            'roi_count=1\nroi_exclude_box=1 "all" 0 0 0 100 100 100 0\n'
            'display_roi_enabled=1\ndisplay_roi_count=1\ndisplay_roi_box=1 "all" 0 0 0 100 100 100 0\n')
        launch(bag);c.click('roi_export');c.click('export_browse');time.sleep(.2)
        c.click('export_aligned');path,manifest=start_export()
        sample=manifest['samples'][0];meta=json.loads((path/sample['metadata']).read_text());points=np.load(path/sample['points'])
        check('bag_imu_leveling_recorded',meta['imu']['applied'] and meta['imu']['axes_corrected'] and manifest['imu_level_requested'])
        check('bag_original_intensity_preserved',np.allclose(np.unique(points[:,3]),[.5,.8]))
        check('bag_source_provenance_recorded',meta['source']['kind']=='rosbag' and meta['source']['topic']=='/b2/points' and
            meta['source']['frame_id']=='pandar_xt32_forward' and meta['source']['timestamp_ns']=='10000000000')
        transform=np.array(meta['sensor_to_output_row_major']).reshape(4,4)
        raw=np.array([-4,-3,-.57,1.])
        check('bag_points_follow_per_frame_imu_transform',np.allclose(points[0,:3],(transform@raw)[:3],atol=2e-6))
        capture(x11,window,out/'03_bag_imu_export.png')
        check('raw_inputs_unchanged',hashlib.sha256((frames/'00000.bin').read_bytes()).hexdigest()==original_hash)
        check('original_profiles_unchanged',before==settings_inventory(repo))
        result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try:capture(x11,window,out/'failure.png')
            except Exception:pass
    finally:
        if process and process.poll() is None:process.terminate();process.wait(timeout=8)
        if x11:x11.close()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

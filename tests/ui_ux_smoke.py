#!/usr/bin/env python3
"""Private Xvfb: fixed navigation, independent scroll, resizing and BIN timeline."""
import argparse,json,os,re,shutil,struct,subprocess,tempfile,time,traceback
from pathlib import Path
from ui_controls import Controls
from ui_gl_smoke import X11,capture,changed_pixels,settings_inventory


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--viewer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 repo=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=True);out=Path(tempfile.mkdtemp(prefix='ux-',dir=a.output.resolve()))
 private=out/'private';(private/'build').mkdir(parents=True);shutil.copytree(repo/'config',private/'config')
 (private/'config/profiles/default.cfg').write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\nroi_count=0\ncalibration_z_m=0\n')
 viewer=private/'build/lidar_viewer';shutil.copy2(a.viewer.resolve(),viewer);clouds=private/'sequence';clouds.mkdir()
 for i in range(12):
  points=[(x*.3,y*.3,0.,.1) for x in range(-20,21) for y in range(-20,21)]
  points.extend((i*.2+x*.1,2+y*.1,z*.1,.9) for x in range(8) for y in range(8) for z in range(15))
  (clouds/f'{i:04d}.bin').write_bytes(b''.join(struct.pack('<4f',*p) for p in points))
 env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),LIDAR_VIEWER_UI_TRACE='1');profile=out/'settings/lidar_viewer/app_settings.cfg'
 before=settings_inventory(repo);result={'status':'started','checks':{},'output':str(out)};process=x=window=None
 def check(label,v):
  result['checks'][label]=bool(v)
  if not v:raise AssertionError(label)
 def prefs():return dict(s.split('=',1) for s in profile.read_text().splitlines() if '=' in s)
 try:
  x=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
  for run in range(2):
   log_path=out/f'viewer-{run}.log'
   with log_path.open('wb') as log:
    process=subprocess.Popen([str(viewer),str(clouds),'--2d','--play'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
    deadline=time.monotonic()+20;window=None
    while not window:
     if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Viewer did not open')
     window=x.viewer_window();time.sleep(.1)
    x.focus(window);x.resize(window,1600,960);c=Controls(x,window,log_path)
    c.rect('tab_View');time.sleep(.15)
    if run==0:
     check('bin_folder_is_one_continuous_sequence','Source frames 0 11' in log_path.read_text())
     first=capture(x,window,out/'01_playing.png');time.sleep(.35);second=capture(x,window,out/'02_motion.png')
     check('bin_sequence_playback_moves',changed_pixels(first,second,(390,180,1540,750))>80)
     x.resize(window,1600,720);c.click('tab_Align');tabs=c.rect('tab_Objects');x.wheel(window,200,500,-20)
     offset=len(log_path.read_text());check('navigation_does_not_scroll_with_settings',c.rect('tab_Objects',after=offset)==tabs)
     c.click('tab_Objects');c.rect('tracking_ids')
     check('can_switch_tabs_after_scrolling',True)
     x.resize(window,1600,960)
     c.click('tab_View');r=c.rect('inspector_resize');start=(int((r[0]+r[2])*.5),int((r[1]+r[3])*.5))
     c.drag(start,(start[0]+110,start[1]));time.sleep(.3)
     check('divider_resize_is_persisted',profile.exists() and 420<=float(prefs()['inspector_width'])<=440)
     resized=capture(x,window,out/'03_wider_panel.png');time.sleep(.35);later=capture(x,window,out/'04_motion_after_resize.png')
     check('resized_canvas_keeps_advancing',changed_pixels(resized,later,(550,180,1540,750))>80)
     c.chord('s');user=out/'settings/lidar_viewer/profiles/default.cfg'
     saved=user if user.exists() else private/'config/profiles/default.cfg'
     check('panel_width_is_not_a_data_setting','inspector_width=' not in saved.read_text())
    else:
     r=c.rect('inspector_resize');check('panel_width_is_restored_on_launch',440<r[0]<455)
     capture(x,window,out/'05_restored_width.png')
    process.terminate();process.wait(timeout=5);process=None;window=None
  check('original_profiles_unchanged',before==settings_inventory(repo));result['status']='passed'
 except Exception as e:
  result.update(status='failed',error=str(e),traceback=traceback.format_exc())
  if process and process.poll() is None and window:
   try:capture(x,window,out/'failure.png')
   except Exception:pass
 finally:
  if process and process.poll() is None:process.terminate();process.wait(timeout=5)
  if x:x.close()
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
 return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())

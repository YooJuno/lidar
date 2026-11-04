#!/usr/bin/env python3
"""Private Xvfb: direct region editing, details dialog, history and visible selection."""
import argparse,json,os,re,shlex,shutil,struct,subprocess,tempfile,time,traceback
from pathlib import Path
from ui_gl_smoke import X11,capture,settings_inventory
from ui_controls import Controls

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--viewer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 repo=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=True);out=Path(tempfile.mkdtemp(prefix='roi-',dir=args.output.resolve()))
 private=out/'private';(private/'build').mkdir(parents=True);shutil.copytree(repo/'config',private/'config')
 profile=private/'config/profiles/default.cfg';profile.write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\nroi_count=0\ncalibration_yaw_deg=0\ncalibration_pitch_deg=0\ncalibration_roll_deg=0\ncalibration_z_m=0\n')
 fixture=private/'cloud.bin';fixture.write_bytes(b''.join(struct.pack('ffff',x,y,z,.5) for x in range(-12,13) for y in range(-12,13) for z in range(-3,2)))
 viewer=private/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
 env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),LIDAR_VIEWER_ROI_TRACE='1',LIDAR_VIEWER_UI_TRACE='1')
 log_path=out/'viewer.log';before=settings_inventory(repo);result={'status':'started','checks':{},'output':str(out)};process=x11=window=c=None
 def check(label,value):
  result['checks'][label]=bool(value)
  if not value:raise AssertionError(label)
 def saved():
  user=out/'settings/lidar_viewer/profiles/default.cfg';return user if user.exists() else profile
 def commit():c.click('roi_apply');c.chord('s');time.sleep(.1)
 def text():return saved().read_text()
 def rois():return '\n'.join(s for s in text().splitlines() if s.startswith('roi_'))
 def inspect():
  r=subprocess.run([str(viewer),str(fixture),'--inspect'],cwd=private,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=15)
  if r.returncode:raise AssertionError(r.stdout)
  return r.stdout
 def count(n,label,boxes):
  commit();s=inspect();check(label,f'{boxes} boxes | retained 3125 / 3125 points' in s and f'Ground ROI: skipped {n} |' in s)
 def edit(center=None,size=None,yaw=None,name=None):
  c.click('roi_details')
  if center is not None:
   for axis,value in enumerate(center):c.enter('roi_center',value,axis)
  if size is not None:
   for axis,value in enumerate(size):c.enter('roi_size',value,axis)
  if yaw is not None:c.enter('roi_yaw',yaw)
  if name is not None:c.enter('roi_name',name)
  c.click('roi_done')
 def passes():return re.findall(r'Analysis input: .*ground passes (\d+)',log_path.read_text(errors='replace'))
 try:
  x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
  with log_path.open('wb') as log:
   process=subprocess.Popen([str(viewer),str(fixture),'--2d'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
   deadline=time.monotonic()+20
   while not window:
    if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Viewer did not open')
    window=x11.viewer_window();time.sleep(.1)
   x11.focus(window);x11.resize(window,1600,960);c=Controls(x11,window,log_path)
   x11.key(window,'r');c.drag((800,300),(1100,570));edit(center=[0,0,-2],size=[4,2,2],yaw=90)
   count(45,'details_position_size_yaw_preserves_full_cloud',1)
   edit(yaw=45);count(21,'yaw_changes_ground_input',1);edit(yaw=90);count(45,'yaw_restore',1)
   c.click('roi_duplicate');count(81,'duplicate_has_offset_and_same_size',2)
   edit(center=[8,0,-2],yaw=0);count(90,'multiple_region_union',2)
   edit(name='nearby');count(90,'rename_keeps_region_geometry',2)
   c.click('roi_delete');count(45,'delete_region',1);c.click('roi_undo');count(90,'undo_delete',2)
   c.click('roi_redo');count(45,'redo_delete',1);c.chord('z');count(90,'keyboard_undo',2)
   c.click('roi_enabled_1');count(45,'disable_one_region',2);c.click('roi_enabled_1');count(90,'reenable_one_region',2)
   capture(x11,window,out/'01_simple_panel.png')
   offset=len(log_path.read_text());time.sleep(.2)
   check('numeric_fields_are_absent_from_main_panel','UI roi_center' not in log_path.read_text()[offset:])
   c.click('roi_add');c.drag((800,300),(1100,570));commit();before_drag=rois()
   check('add_region_starts_drawing','roi_count=3' in before_drag)
   c.drag((1100,570),(1140,600));commit();check('corner_resize_changes_box',rois()!=before_drag)
   c.chord('z');commit();check('undo_resize_restores_box',rois()==before_drag)
   c.drag((950,270),(1115,435));commit();last=[r for r in text().splitlines() if r.startswith('roi_exclude_box=')][-1]
   check('round_handle_rotates_box',abs(abs(float(last.split()[-1]))-90)<2)
   c.chord('z');commit();check('undo_rotation_restores_box',rois()==before_drag)
   c.drag((950,420),(1000,450));commit();check('drag_inside_moves_box',rois()!=before_drag)
   c.chord('z');commit();check('undo_move_restores_box',rois()==before_drag)
   x11.key(window,'r');x11.move(window,700,250);x11.xt.XTestFakeButtonEvent(x11.display,1,True,0);x11.move(window,800,350);x11.key(window,'Escape');x11.xt.XTestFakeButtonEvent(x11.display,1,False,0);x11.x.XSync(x11.display,False);time.sleep(.2)
   commit();check('escape_cancels_drawing',rois()==before_drag)
   x11.click(window,1300,700)
   for _ in range(3):x11.key(window,'e')
   x11.key(window,'r');c.drag((800,300),(1100,570));commit();last=[r for r in text().splitlines() if r.startswith('roi_exclude_box=')][-1]
   check('draw_follows_rotated_view',abs(float(last.split()[-1])-15)<.1)
   c.chord('z');commit();check('undo_rotated_draw_restores_regions',rois()==before_drag)
   x11.click(window,155,140);x11.click(window,46,222);time.sleep(.3);x11.key(window,'r');x11.key(window,'Escape')
   old_passes=passes();edit(center=[1,0,-2]);edit(center=[2,0,-2]);check('draft_edit_does_not_resegment',passes()==old_passes)
   commit();check('apply_resegments_ground',bool(passes() and old_passes and int(passes()[-1])>int(old_passes[-1])))
   for _ in range(6):c.click('roi_duplicate')
   commit();row=c.rect('roi_row_8');listing=c.rect('roi_list')
   check('selected_ninth_region_is_visible',row[1]>=listing[1] and row[3]<=listing[3])
   capture(x11,window,out/'02_nine_regions.png')
   def names(display=False):
    prefix='display_roi_box=' if display else 'roi_exclude_box='
    return [shlex.split(line.split('=',1)[1])[1] for line in text().splitlines() if line.startswith(prefix)]
   def deletion_order(display=False):
    label='display_' if display else 'ground_'
    def persist():
     if not display:c.click('roi_apply')
     c.chord('s');time.sleep(.1)
    original=names(display);check(label+'delete_fixture_has_nine_regions',len(original)==9)
    c.click('roi_row_6');c.click('roi_enabled_8');c.click('roi_enabled_8');c.click('roi_delete');persist()
    check(label+'delete_removes_selected_middle_only',names(display)==original[:6]+original[7:])
    c.click('roi_delete');persist()
    check(label+'delete_continues_with_next_region',names(display)==original[:6]+original[8:])
    c.click('roi_undo');persist();c.click('roi_undo');persist()
    check(label+'undo_both_deletions_restores_regions',names(display)==original)
    x,y,r,b=c.rect('roi_list')
    for _ in range(3):x11.click(window,int((x+r)*.5),int((y+b)*.5),5)
    time.sleep(.2);c.click('roi_row_8');x11.key(window,'Delete');persist()
    check(label+'keyboard_delete_removes_last_region',names(display)==original[:-1])
    x11.key(window,'Delete');persist()
    check(label+'last_delete_selects_previous_region',names(display)==original[:-2])
    for _ in range(7):c.click('roi_delete');persist()
    check(label+'can_delete_through_last_remaining_region',names(display)==[])
    c.click('roi_undo');persist()
    check(label+'undo_restores_last_remaining_region',names(display)==original[:1])
   deletion_order()
   ground_names=names();c.click('roi_purpose_display');c.click('roi_add');c.drag((800,300),(1100,570))
   for _ in range(8):c.click('roi_duplicate')
   c.chord('s');time.sleep(.1);deletion_order(True)
   check('display_deletions_preserve_ground_regions',names()==ground_names)
   c.click('display_roi_enabled');c.chord('s')
   c.click('roi_purpose_ground');capture(x11,window,out/'03_deletion_selection.png')
   base='\n'.join(line for line in text().splitlines() if not line.startswith('roi_'))+'\n'
   base+='roi_enabled=0\nroi_count=2\nroi_box=1 "a" 0 0 -2 4 2 2 90\nroi_exclude_box=1 "b" 8 0 -2 4 2 2 0\ncalibration_yaw_deg=19\ncalibration_pitch_deg=-12\ncalibration_roll_deg=9\ncalibration_z_m=.57\n'
   for ground in [0,1]:
    saved().write_text(base+f'ground_removal_enabled={ground}\nground_mode=fast_grid\n');s=inspect()
    check('pipeline_transform_ground_'+str(ground),'2 boxes | retained 3125 / 3125 points' in s and 'Ground ROI: skipped 90 | input 3035' in s)
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

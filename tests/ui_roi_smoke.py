#!/usr/bin/env python3
"""Private Xvfb: direct region editing, details dialog, history and visible selection."""
import argparse,json,math,os,re,shlex,shutil,struct,subprocess,tempfile,time,traceback
from pathlib import Path
from ui_gl_smoke import X11,capture,changed_pixels,settings_inventory
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
  c.roi_action('details')
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
   x11.key(window,'r');c.drag((800,300),(1100,570))
   c.right_click('roi_selected_row');capture(x11,window,out/'00_context_menu.png');c.click('roi_details')
   widths=[]
   for _ in range(5):
    offset=len(log_path.read_text());time.sleep(.2);x,y,r,b=c.rect('roi_name',after=offset);widths.append(r-x)
   check('details_width_stays_fixed_when_idle',max(widths)-min(widths)<1.5 and min(widths)>350)
   capture(x11,window,out/'00_details.png');c.click('roi_done')
   edit(center=[0,0,-2],size=[4,2,2],yaw=90)
   count(45,'details_position_size_yaw_preserves_full_cloud',1)
   edit(yaw=45);count(21,'yaw_changes_ground_input',1);edit(yaw=90);count(45,'yaw_restore',1)
   c.roi_action('duplicate');count(81,'duplicate_has_offset_and_same_size',2)
   edit(center=[8,0,-2],yaw=0);count(90,'multiple_region_union',2)
   edit(name='nearby');count(90,'rename_keeps_region_geometry',2)
   c.roi_action('rename',0);c.enter('roi_rename_name','train front');commit()
   def region_names(display=False):
    prefix='display_roi_box='if display else'roi_exclude_box='
    return[shlex.split(line.split('=',1)[1])[1]for line in text().splitlines()if line.startswith(prefix)]
   check('list_rename_targets_clicked_region_only',region_names()==['train front','nearby'])
   c.chord('z');commit();check('list_rename_undo_restores_name',region_names()==['ROI 1','nearby'])
   c.chord('y');commit();check('list_rename_redo_restores_name',region_names()==['train front','nearby'])
   c.roi_action('rename',1);c.enter('roi_rename_name','');c.click('roi_rename_cancel');commit()
   check('empty_name_and_cancel_keep_region_unchanged',region_names()==['train front','nearby'])
   count(90,'list_rename_preserves_geometry_and_ground_input',2)
   c.click('roi_row_1')
   c.roi_action('details');c.enter('roi_bottom',-3.5);c.enter('roi_top',-.5);c.click('roi_done');commit()
   height=shlex.split([line for line in text().splitlines() if line.startswith('roi_exclude_box=')][-1].split('=',1)[1])
   check('height_bounds_accept_direct_input',abs(float(height[4])+2)<1e-5 and abs(float(height[7])-3)<1e-5)
   edit(center=[8,0,-2],size=[4,2,2]);count(90,'height_edit_preserves_other_box_dimensions',2)
   c.roi_action('delete');count(45,'delete_region',1);c.chord('z');count(90,'undo_delete',2)
   c.chord('y');count(45,'redo_delete',1);c.chord('z');count(90,'keyboard_undo',2)
   c.click('roi_enabled_1');count(45,'disable_one_region',2);c.click('roi_enabled_1');count(90,'reenable_one_region',2)
   visible=capture(x11,window,out/'enabled_regions.png')
   c.click('roi_enabled_0');c.click('roi_enabled_1');x11.move(window,200,140)
   disabled=capture(x11,window,out/'disabled_regions.png')
   c.roi_toggle_outlines();x11.move(window,200,140)
   hidden=capture(x11,window,out/'outlines_hidden.png')
   check('disabled_regions_have_no_outlines_or_handles',changed_pixels(disabled,hidden,(380,180,1540,750))==0)
   check('enabled_regions_have_visible_outlines',changed_pixels(visible,disabled,(380,180,1540,750))>30)
   c.roi_toggle_outlines();c.click('roi_enabled_0');c.click('roi_enabled_1');commit()
   capture(x11,window,out/'01_simple_panel.png')
   offset=len(log_path.read_text());time.sleep(.2)
   check('numeric_fields_are_absent_from_main_panel','UI roi_center' not in log_path.read_text()[offset:])
   check('editing_buttons_are_absent_from_main_panel',not any('UI '+name+' 'in log_path.read_text()[offset:]for name in
    ['roi_focus','roi_duplicate','roi_delete','roi_rename_0','roi_undo','roi_redo','roi_outlines','roi_bottom','roi_top']))
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
   for _ in range(6):c.roi_action('duplicate')
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
    c.click('roi_row_8');c.roi_action('delete',6);persist()
    check(label+'right_click_delete_targets_clicked_row',names(display)==original[:6]+original[7:])
    c.chord('z');persist()
    c.click('roi_row_6');c.click('roi_enabled_8');c.click('roi_enabled_8');c.roi_action('delete');persist()
    check(label+'delete_removes_selected_middle_only',names(display)==original[:6]+original[7:])
    c.roi_action('delete');persist()
    check(label+'delete_continues_with_next_region',names(display)==original[:6]+original[8:])
    c.chord('z');persist();c.chord('z');persist()
    check(label+'undo_both_deletions_restores_regions',names(display)==original)
    x,y,r,b=c.rect('roi_list')
    for _ in range(3):x11.click(window,int((x+r)*.5),int((y+b)*.5),5)
    time.sleep(.2);c.click('roi_row_8');x11.key(window,'Delete');persist()
    check(label+'keyboard_delete_removes_last_region',names(display)==original[:-1])
    x11.key(window,'Delete');persist()
    check(label+'last_delete_selects_previous_region',names(display)==original[:-2])
    for _ in range(7):c.roi_action('delete');persist()
    check(label+'can_delete_through_last_remaining_region',names(display)==[])
    c.chord('z');persist()
    check(label+'undo_restores_last_remaining_region',names(display)==original[:1])
   deletion_order()
   ground_names=names();c.click('roi_purpose_display');c.click('roi_add');c.drag((800,300),(1100,570))
   for _ in range(8):c.roi_action('duplicate')
   c.roi_action('rename',8);c.enter('roi_rename_name','person');c.chord('s');time.sleep(.1)
   check('display_region_list_rename_is_saved',region_names(True)[-1]=='person')
   c.chord('s');time.sleep(.1);deletion_order(True)
   check('display_deletions_preserve_ground_regions',names()==ground_names)
   def camera():
    c.chord('s');time.sleep(.1)
    return {k:float(v) for line in text().splitlines() if '=' in line for k,v in [line.split('=',1)]
      if k in ['camera_distance','camera_target_x','camera_target_y','camera_target_z','camera_yaw_rad']}
   edit(center=[1,0,-2],size=[.08,.08,.1],yaw=0);c.double_click('roi_selected_row');zoom=camera()
   check('focus_small_roi_can_zoom_below_old_limit',zoom['camera_distance']<.2)
   x11.wheel(window,968,431,4);zoom=camera()
   check('2d_zoom_reaches_twenty_centimeter_view',abs(zoom['camera_distance']-.1)<1e-5)
   c.drag((1300,620),(1336,638));panned=camera();unit=.2/718;co=math.cos(zoom['camera_yaw_rad']);si=math.sin(zoom['camera_yaw_rad'])
   check('high_zoom_panning_tracks_screen_distance',abs(panned['camera_target_x']-zoom['camera_target_x']+(36*co+18*si)*unit)<1e-5 and
     abs(panned['camera_target_y']-zoom['camera_target_y']-(-36*si+18*co)*unit)<1e-5)
   c.double_click('roi_selected_row');zoom=camera();x11.wheel(window,1120,500,2);anchored=camera()
   check('zoom_keeps_cursor_world_position',
     abs((zoom['camera_target_x']+(152*co+69*si)*2*zoom['camera_distance']/718)-
       (anchored['camera_target_x']+(152*co+69*si)*2*anchored['camera_distance']/718))<1e-5 and
     abs((zoom['camera_target_y']+(152*si-69*co)*2*zoom['camera_distance']/718)-
       (anchored['camera_target_y']+(152*si-69*co)*2*anchored['camera_distance']/718))<1e-5)
   c.double_click('roi_selected_row');x11.wheel(window,968,431,4);c.drag((968,431),(1004,449));c.chord('s')
   box=shlex.split([line for line in text().splitlines() if line.startswith('display_roi_box=')][0].split('=',1)[1])
   check('roi_move_matches_magnified_cloud_coordinates',abs(float(box[2])-(1+(36*co+18*si)*unit))<1e-5 and
     abs(float(box[3])-(36*si-18*co)*unit)<1e-5)
   capture(x11,window,out/'04_close_zoom.png');zoom=camera()
   c.click('camera_3D');c.click('camera_2D');restored=camera()
   check('projection_round_trip_preserves_close_zoom',all(abs(zoom[k]-restored[k])<1e-5 for k in zoom))
   c.click('display_roi_enabled');c.chord('s')
   c.click('roi_purpose_ground');capture(x11,window,out/'03_deletion_selection.png')
   base='\n'.join(line for line in text().splitlines() if not line.startswith('roi_'))+'\n'
   base+='roi_enabled=0\nroi_count=2\nroi_box=1 "a" 0 0 -2 4 2 2 90\nroi_exclude_box=1 "b" 8 0 -2 4 2 2 0\ncalibration_yaw_deg=19\ncalibration_pitch_deg=-12\ncalibration_roll_deg=9\ncalibration_z_m=.57\n'
   for ground in [0,1]:
    saved().write_text(base+f'ground_removal_enabled={ground}\nground_mode=fast_grid\n');s=inspect()
    check('pipeline_transform_ground_'+str(ground),'2 boxes | retained 3125 / 3125 points' in s and 'Ground ROI: skipped 90 | input 3035' in s)
   process.terminate();process.wait(timeout=5);window=None
   with log_path.open('ab') as log:process=subprocess.Popen([str(viewer),str(fixture),'--2d'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
   deadline=time.monotonic()+20
   while not window:
    if process.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Viewer did not reopen')
    window=x11.viewer_window();time.sleep(.1)
   x11.focus(window);x11.resize(window,1600,960);c=Controls(x11,window,log_path);time.sleep(.3)
   check('saved_close_zoom_survives_restart',abs(camera()['camera_distance']-.1)<1e-5)
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

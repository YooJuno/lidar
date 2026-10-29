#!/usr/bin/env python3
"""Private Xvfb, real large bag: visible progress, cancel, recovery and preserved playback."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import traceback
from ui_gl_smoke import X11,capture,changed_pixels,settings_inventory


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--bag',type=Path,required=True)
    parser.add_argument('--small-bag',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True,exist_ok=True)
    output=Path(tempfile.mkdtemp(prefix='loading-',dir=args.output.resolve()))
    private=output/'private';(private/'build').mkdir(parents=True);shutil.copytree(repo/'config',private/'config')
    (private/'config/profiles/rosbag.cfg').write_text('ground_removal_enabled=0\nclustering_enabled=0\nroi_enabled=0\nroi_count=0\nimu_level_enabled=0\n')
    viewer=private/'build/lidar_viewer';shutil.copy2(args.viewer.resolve(),viewer)
    env=dict(os.environ,XDG_CONFIG_HOME=str(output/'isolated'),GSETTINGS_BACKEND='memory',NO_AT_BRIDGE='1')
    before=settings_inventory(repo);process=x11=window=None;deadline=time.monotonic()+150
    result={'status':'started','checks':{},'output':str(output)};log_path=output/'viewer.log'
    def wait_for(predicate):
        while not predicate():
            if process.poll() is not None:raise RuntimeError('Viewer exited unexpectedly')
            if time.monotonic()>deadline:raise TimeoutError('Loading UI timed out')
            time.sleep(.04)
    def text():return log_path.read_text(errors='replace')
    def check(label,value):
        result['checks'][label]=bool(value)
        if not value:raise AssertionError(label)
    def event(key,down):
        code=x11.x.XKeysymToKeycode(x11.display,x11.x.XStringToKeysym(key.encode()));x11.xt.XTestFakeKeyEvent(x11.display,code,down,0)
    def chord(key):
        event('Control_L',True);event(key,True);event(key,False);event('Control_L',False);x11.x.XSync(x11.display,False);time.sleep(.15)
    def chooser():return next((w for w in x11.children(x11.root) if x11.title(w)=='Open LiDAR data'),None)
    def choose(path):
        x11.focus(window);chord('o');wait_for(chooser);dialog=chooser();x11.focus(dialog);chord('l');chord('a')
        for ch in str(path):
            shifted=ch.isupper() or ch=='_'
            key={'/':'slash','_':'underscore','.':'period','-':'minus',' ':'space'}.get(ch,ch.lower())
            if shifted:event('Shift_L',True)
            event(key,True);event(key,False)
            if shifted:event('Shift_L',False)
        x11.x.XSync(x11.display,False);time.sleep(.4)
        event('Return',True);event('Return',False);x11.x.XSync(x11.display,False)
        # GTK can first navigate to a typed absolute path, then enable Open
        # once its asynchronous directory query completes.
        time.sleep(.8)
        if chooser():x11.focus(chooser());x11.key(chooser(),'Return')
        dialog_deadline=time.monotonic()+12
        while chooser():
            if time.monotonic()>dialog_deadline:
                capture(x11,chooser(),output/'chooser-failure.png')
                raise TimeoutError('Native chooser did not accept the selected path')
            time.sleep(.05)
        x11.focus(window)
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with log_path.open('wb') as log:
            process=subprocess.Popen([str(viewer),str(args.bag.resolve()),'--2d'],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
            wait_for(x11.viewer_window);window=x11.viewer_window();x11.focus(window);x11.resize(window,1600,960)
            wait_for(lambda:'indexed 1000' in text())
            check('window_opens_during_indexing','Opened bag:' not in text())
            a=capture(x11,window,output/'01_progress.png');time.sleep(.4);b=capture(x11,window,output/'02_progress_advances.png')
            check('loading_card_updates_while_indexing',changed_pixels(a,b,(708,310,1228,555))>30)
            x11.key(window,'Escape');wait_for(lambda:'Opening canceled; current input preserved' in text())
            check('cancel_keeps_empty_viewer_open',process.poll() is None and 'Input could not be opened:' not in text())
            bad=output/'broken.mcap';bad.write_bytes(b'not a recording')
            choose(bad);wait_for(lambda:'Input could not be opened:' in text())
            capture(x11,window,output/'03_failed_open.png')
            choose(args.small_bag.resolve()/'metadata.yaml');wait_for(lambda:'Opened bag:' in text());time.sleep(.2)
            check('valid_bag_opens_after_cancel_and_error',process.poll() is None)
            x11.click(window,410,837);playing=capture(x11,window,output/'04_playing.png')
            choose(args.bag.resolve()/'metadata.yaml');wait_for(lambda:text().count('Indexing ROS 2 bag')>=3)
            capture(x11,window,output/'05_switch_progress.png')
            x11.key(window,'Escape');wait_for(lambda:text().count('Opening canceled; current input preserved')>=2)
            time.sleep(.3);resumed=capture(x11,window,output/'06_preserved_playback.png')
            check('canceled_replacement_restores_previous_playback',changed_pixels(playing,resumed,(1235,20,1320,48))==0)
            check('canceled_replacement_does_not_replace_input',text().count('Opened bag:')==1)
            check('original_profiles_unchanged',before==settings_inventory(repo));result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and window and process and process.poll() is None:
            try:capture(x11,window,output/'failure.png')
            except Exception:pass
    finally:
        if process and process.poll() is None:
            # Escape cancels an outstanding open, then the normal quit path can finish.
            if x11 and window:
                x11.key(window,'Escape');time.sleep(.3);x11.key(window,'Escape')
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=5)
        if x11:x11.close()
        (output/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())

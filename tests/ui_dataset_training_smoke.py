#!/usr/bin/env python3
"""Private Xvfb: dataset reopening/navigation and owned training-job controls."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import numpy as np
from ui_gl_smoke import X11,capture,settings_inventory
from ui_controls import Controls
from training_adapter_smoke import fixture


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--viewer',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];args.output.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='workspace-',dir=args.output.resolve()));private=out/'private';(private/'build').mkdir(parents=True)
    shutil.copy2(args.viewer.resolve(),private/'build/lidar_studio');shutil.copytree(repo/'config',private/'config');shutil.copytree(repo/'python',private/'python',ignore=shutil.ignore_patterns('__pycache__'))
    (private/'config/profiles/rosbag.cfg').write_text('ground_removal_enabled=0\nclustering_enabled=0\ndetection_enabled=0\nroi_count=0\n')
    data=out/'dataset';fixture(data);project=out/"trainer 'literal'";(project/'training').mkdir(parents=True);(project/'training/__init__.py').touch();(project/'training/litept_adapter.py').touch()
    (project/'torch.py').write_text('class cuda:\n @staticmethod\n def is_available(): return True\n')
    (project/'base_config.py').write_text("model=dict(type='CenterPoint')\n")
    (project/'training/run.py').write_text("import os,argparse,json,time\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--epochs',type=int);a,rest=p.parse_known_args();out=Path(a.output);out.mkdir();(out/'contract.json').write_text(json.dumps(dict(epochs=a.epochs,args=rest,pid=os.getpid())));print('TRAINER CONTRACT STARTED',flush=True)\nfor epoch in range(a.epochs):\n time.sleep(.03 if a.epochs<10 else 1);print(json.dumps(dict(epoch=epoch+1,train_loss=.5/(epoch+1))),flush=True)\n(out/'contract_complete.txt').write_text('process contract only; no model accuracy claimed')\n")
    settings=out/'settings/lidar_viewer';settings.mkdir(parents=True);runs=out/'runs'
    profiles={name:dict(python=sys.executable,project=str(project),dataset=str(data),config=str(project/'base_config.py'),output=str(runs),epochs=2,max_points=128,seed=42,classes='person,car')for name in('litept','centerpoint')}
    (settings/'training.json').write_text(json.dumps(dict(selected=0,profiles=profiles)))
    chooser=out/'chooser';chooser.mkdir();(chooser/'zenity').write_text('#!'+sys.executable+'\nimport os\nprint(os.environ["DATASET_FOLDER"])\n');(chooser/'zenity').chmod(0o755)
    env=dict(os.environ,XDG_CONFIG_HOME=str(out/'settings'),PATH=str(chooser)+os.pathsep+os.environ['PATH'],DATASET_FOLDER=str(data),LIDAR_VIEWER_UI_TRACE='1')
    before=settings_inventory(repo);log_path=out/'viewer.log';result=dict(status='started',checks={},output=str(out));process=x11=window=controls=None
    def check(name,condition):
        result['checks'][name]=bool(condition)
        if not condition:raise AssertionError(name)
    def wait(condition,timeout=20):
        end=time.monotonic()+timeout
        while not condition():
            if process.poll()is not None:raise RuntimeError(log_path.read_text(errors='replace')[-2500:])
            if time.monotonic()>end:raise TimeoutError('Dataset/training GUI timed out')
            time.sleep(.05)
    def jobs():return sorted(runs.glob('litept_*/result.json'))if runs.exists()else[]
    try:
        x11=X11();subprocess.run(['xrdb','-merge'],input='Xft.dpi: 96\n',text=True,check=True)
        with log_path.open('wb')as log:process=subprocess.Popen([str(private/'build/lidar_studio')],cwd=private,env=env,stdout=log,stderr=subprocess.STDOUT)
        wait(lambda:x11.viewer_window()is not None);window=x11.viewer_window();x11.focus(window);x11.resize(window,1600,960);controls=Controls(x11,window,log_path)
        controls.click('open_dataset');wait(lambda:'Opened point cloud:'in log_path.read_text());controls.rect('dataset_current_0')
        check('dataset_folder_opens_in_native_browser',True);capture(x11,window,out/'01_dataset.png')
        offset=len(log_path.read_text());x11.key(window,'Right');controls.rect('dataset_current_1',after=offset)
        check('right_arrow_advances_dataset_sample',True);offset=len(log_path.read_text());x11.key(window,'Left');controls.rect('dataset_current_0',after=offset)
        check('left_arrow_returns_to_first_sample',True);controls.chord('s');time.sleep(.2)
        profile=out/'settings/lidar_viewer/profiles/dataset.cfg';values=dict(line.split('=',1)for line in profile.read_text().splitlines()if '='in line)
        check('dataset_view_does_not_apply_sensor_height_or_imu',float(values['calibration_z_m'])==0 and values['imu_level_enabled']=='0'and values['ground_removal_enabled']=='0')
        controls.click('dataset_train');controls.rect('training_epochs');controls.click('training_validate');wait(lambda:len(jobs())==1)
        check('training_setup_validates_annotated_dataset',json.loads(jobs()[0].read_text())['status']=='validated')
        controls.click('training_start');wait(lambda:len(jobs())==2)
        check('training_uses_selected_epochs_and_writes_run_status',json.loads(jobs()[1].read_text())['status']=='complete'and json.loads((jobs()[1].parent/'model/contract.json').read_text())['epochs']==2)
        capture(x11,window,out/'02_training.png');controls.enter('training_epochs',100);controls.click('training_start')
        wait(lambda:any(runs.glob('litept_003/model/contract.json')));controls.click('training_hide')
        offset=len(log_path.read_text());x11.key(window,'Right');controls.rect('dataset_current_1',after=offset)
        check('dataset_navigation_remains_responsive_during_training',True);controls.click('open_training');controls.click('training_stop');time.sleep(.4)
        check('stop_leaves_incomplete_job_without_success_marker',not(runs/'litept_003/result.json').exists()and not(runs/'litept_003/model/contract_complete.txt').exists())
        controls.click('training_adapter');controls.click('training_adapter_centerpoint');controls.rect('training_batch_size');controls.rect('training_learning_rate')
        check('centerpoint_has_its_own_parameters',True);capture(x11,window,out/'03_centerpoint_settings.png')
        check('original_profiles_unchanged',before==settings_inventory(repo));result['status']='passed'
    except Exception as error:
        result.update(status='failed',error=str(error),traceback=traceback.format_exc())
        if x11 and window and process and process.poll()is None:
            try:capture(x11,window,out/'failure.png')
            except Exception:pass
    finally:
        if process and process.poll()is None:process.terminate();process.wait(timeout=5)
        if x11:x11.close()
        (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
    return 0 if result['status']=='passed'else 1


if __name__=='__main__':raise SystemExit(main())

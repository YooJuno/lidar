#!/usr/bin/env python3
"""Dataset conversion and trainer CLI contracts; no learned predictions are mocked as accuracy."""
import json
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
from lidar_training.data import samples, points, labels, split, prepare_litept
from lidar_training.centerpoint import prepare, annotations, configure_head, configure_schedule


def reject(action):
    try:action()
    except(ValueError,FileNotFoundError):return
    raise AssertionError('Invalid annotations were accepted')


def fixture(root):
    for n in range(2):
        directory=root/f'group{n}';(directory/'points').mkdir(parents=True);(directory/'metadata').mkdir();(directory/'labels').mkdir();(directory/'annotations').mkdir()
        cloud=np.array([[1+n,0,0,42],[1+n,.1,0,84],[1+n,.2,.1,128]],dtype='<f4');np.save(directory/'points/sample.npy',cloud)
        np.save(directory/'labels/sample.npy',np.array([0,1,2],dtype=np.int32))
        (directory/'metadata/sample.json').write_text(json.dumps(dict(source=dict(path=f'bag{n}',frame_number=10),coordinate_space='sensor')))
        (directory/'annotations/sample.json').write_text(json.dumps(dict(coordinate_space='sensor',boxes=[dict(class_name='person',center_m=[1+n,0,0],size_m=[.5,.6,1.7],yaw_deg=90)])))
        (directory/'metadata.json').write_text(json.dumps(dict(schema='lidar_studio.roi_dataset',status='complete',samples=[dict(id='sample',points='points/sample.npy',metadata='metadata/sample.json')])) )


def main():
    with tempfile.TemporaryDirectory(prefix='studio-training-')as temporary:
        root=Path(temporary);data=root/'data';fixture(data);fixture(data/'.lidar_studio_trash');records=list(samples(data));assert len(records)==2
        train,val=split(records,.2,42);assert set(train).isdisjoint(val)and len(train)==len(val)==1
        manifest=prepare_litept(data,root/'litept-prepared',.2,42,False);info=json.loads(manifest.read_text())
        with np.load(info['context_npz'],allow_pickle=False)as arrays:
            assert arrays['coord'].shape==(6,3)and np.array_equal(arrays['intensity'],[42,84,128,42,84,128])
            assert np.array_equal(arrays['labels'],[0,1,2,0,1,2])and np.array_equal(arrays['offsets'],[0,3,6])
        request=dict(dataset=str(data),classes='person,car',validation_fraction=.2,seed=42)
        classes=prepare(request,root/'centerpoint-prepared',5);assert classes==['person','car']
        with(root/'centerpoint-prepared/infos_train.pkl').open('rb')as file:prepared=pickle.load(file)
        box=prepared['data_list'][0]['instances'][0]['bbox_3d'];assert len(box)==9 and abs(box[6]-np.pi/2)<1e-6 and box[5]==1.7
        model=dict(pts_bbox_head=dict(tasks=[dict(class_names=['car']),dict(class_names=['pedestrian'])],common_heads=dict(vel=(2,2))),train_cfg=dict(pts=dict(code_weights=[1.]*10)),test_cfg=dict(pts=dict(min_radius=[4.,.175])))
        configure_head(model,['person','car'])
        assert model['train_cfg']['pts']['code_weights'][-2:]==[0.,0.]and model['test_cfg']['pts']['min_radius']==[.175,4.]
        class Config(dict):
            __getattr__=dict.__getitem__
            __setattr__=dict.__setitem__
        schedule=Config(train_cfg=dict(max_epochs=20),optim_wrapper=dict(optimizer=dict(lr=.0001)),param_scheduler=[dict(type='CosineAnnealingLR',T_max=12,eta_min=.00000001,begin=8,end=20,by_epoch=True)])
        configure_schedule(schedule,dict(epochs=30,learning_rate=.001))
        assert schedule.param_scheduler[0]['begin']==12 and schedule.param_scheduler[0]['end']==30 and schedule.param_scheduler[0]['T_max']==18 and abs(schedule.param_scheduler[0]['eta_min']-1e-7)<1e-12
        cloud=np.fromfile(prepared['data_list'][0]['lidar_points']['lidar_path'],dtype='<f4').reshape(-1,5);assert np.all(cloud[:,4]==0)and np.array_equal(cloud[:,3],[42,84,128])
        sample=records[0];np.save(sample['labels'],np.full(3,-1,dtype=np.int32));reject(lambda:labels(sample,3))
        np.save(sample['labels'].parent/'weak_segment.npy',np.array([0,1,1],dtype=np.int32));assert labels(sample,3,True)[1]
        sample['boxes'].write_text(json.dumps(dict(boxes=[dict(class_name='person',center_m=[0,0,0],size_m=[1,-1,1],yaw_deg=0)])));reject(lambda:annotations(sample,['person']))
        # Exercise the real adapter's dispatch/progress/error protocol with an
        # isolated trainer contract fixture, not a stand-in detection model.
        project=root/'trainer';(project/'training').mkdir(parents=True);(project/'training/__init__.py').touch();(project/'training/litept_adapter.py').touch()
        (project/'torch.py').write_text('class cuda:\n @staticmethod\n def is_available(): return True\n')
        (project/'training/run.py').write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--output');a,rest=p.parse_known_args();out=Path(a.output);out.mkdir();(out/'contract_args.json').write_text(json.dumps(rest));print(json.dumps(dict(epoch=1,train_loss=.25)),flush=True)\n")
        run=root/'run';run.mkdir();job=dict(adapter='litept',project=str(project),dataset=str(manifest),epochs=1,max_points=128,seed=42,allow_weak=False)
        path=run/'request.json';path.write_text(json.dumps(job));done=subprocess.run([sys.executable,str(ROOT/'python/studio_train.py'),'--request',str(path)],capture_output=True,text=True)
        assert done.returncode==0,done.stdout+done.stderr
        assert json.loads((run/'result.json').read_text())['status']=='complete'and json.loads((run/'progress.json').read_text())['epoch']==1
        args=json.loads((run/'model/contract_args.json').read_text());assert args[args.index('--epochs')+1]=='1'and args[args.index('--max-points')+1]=='128'
        print('Training adapters: point/box formats, disjoint splits, weak-label opt-in, coordinates, CLI dispatch and progress passed')


if __name__=='__main__':main()

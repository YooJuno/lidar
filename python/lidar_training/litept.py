"""Adapter for the inspected lidar_segmentation training.run CLI."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from .data import prepare_litept, samples, points, labels, split


def validate(request):
    project=Path(request['project']).expanduser().resolve();dataset=Path(request['dataset']).expanduser().resolve()
    if not(project/'training/run.py').is_file()or not(project/'training/litept_adapter.py').is_file():raise ValueError('Select the LitePT training project containing training/run.py and training/litept_adapter.py')
    if importlib.util.find_spec('torch')is None:raise ValueError('Selected Python has no PyTorch; use the project training environment')
    import torch
    if not torch.cuda.is_available():raise ValueError('LitePT requires a CUDA PyTorch training environment')
    if dataset.is_file():
        manifest=json.loads(dataset.read_text());source=Path(manifest.get('context_npz',''))
        if not source.is_absolute():source=project/source
        if not source.is_file():raise ValueError('Prepared manifest context_npz is missing')
        if 'weak'in manifest.get('label_type','').lower()and not request['allow_weak']:raise ValueError('This split uses weak ROI labels. Enable weak labels explicitly, or choose manually annotated data')
        train=manifest.get('train_indices',[]);val=manifest.get('val_indices',[])
        if not train or not val or set(train)&set(val):raise ValueError('Prepared split needs disjoint, nonempty train/validation indices')
        with np.load(source,allow_pickle=False)as arrays:
            if any(k not in arrays for k in ('coord','intensity','labels','offsets')):raise ValueError('Prepared arrays are incomplete')
            coord=arrays['coord'];intensity=arrays['intensity'];target=arrays['labels'];offsets=arrays['offsets']
            if coord.ndim!=2 or coord.shape[1]!=3 or coord.dtype.kind!='f' or not np.isfinite(coord).all():raise ValueError('Prepared coord must be finite floating N×3')
            if intensity.shape!=(len(coord),)or not np.isfinite(intensity).all():raise ValueError('Prepared intensity must match coordinates')
            if target.shape!=(len(coord),)or target.dtype.kind not in 'iu' or target.min(initial=-1)<-1 or target.max(initial=0)>2:raise ValueError('Prepared labels must be N integers -1,0,1,2')
            if offsets.ndim!=1 or offsets.dtype.kind not in 'iu' or len(offsets)<3 or offsets[0]!=0 or offsets[-1]!=len(coord)or not(np.diff(offsets)>0).all():raise ValueError('Prepared offsets must partition nonempty samples')
            if any(type(i)is not int or i<0 or i>=len(offsets)-1 for i in train+val):raise ValueError('Prepared split index is out of bounds')
            known=set()
            for i in train:known.update(np.unique(target[offsets[i]:offsets[i+1]]).tolist())
            if 0 not in known or not({1,2}&known):raise ValueError('Prepared training needs background and foreground labels')
    else:
        records=list(samples(dataset));train,_=split(records,request['validation_fraction'],request['seed']);known=set()
        for index,sample in enumerate(records):
            target,_=labels(sample,len(points(sample)),request['allow_weak'])
            if index in train:known.update(np.unique(target).tolist())
        if 0 not in known or not({1,2}&known):raise ValueError('Training needs labeled background and foreground context; positive-only ROI crops are insufficient')
    return dict(project=str(project),dataset=str(dataset),parameters='epochs, max points, yaw augmentation and seed; optimizer/batch follow the project recipe')


def run(request, output):
    dataset=Path(request['dataset']).expanduser().resolve()
    if dataset.is_dir():dataset=prepare_litept(dataset,output/'prepared',request['validation_fraction'],request['seed'],request['allow_weak'])
    command=[sys.executable,'-u','-m','training.run','--model','litept','--recipe','robust_v1',
        '--epochs',str(request['epochs']),'--max-points',str(request['max_points']),'--seed',str(request['seed']),
        '--yaw-range-deg',str(request['yaw_range']),'--manifest',str(dataset),'--output',str(output/'model')]
    print('Training command:',json.dumps(command,ensure_ascii=False),flush=True)
    with subprocess.Popen(command,cwd=request['project'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)as child:
        for line in child.stdout:
            print(line,end='',flush=True)
            try:
                metric=json.loads(line)
                if isinstance(metric,dict)and'epoch'in metric:
                    pending=output/'progress.tmp';pending.write_text(json.dumps(metric));pending.replace(output/'progress.json')
            except(ValueError,TypeError):pass
        if child.wait():raise RuntimeError(f'LitePT trainer exited with code {child.returncode}')

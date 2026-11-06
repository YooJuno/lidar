"""Read annotated exports without guessing labels from ROI names or model predictions."""
import hashlib
import fnmatch
import os
import json
from pathlib import Path
import numpy as np


def inside(root, name):
    path=(root/name).resolve()
    if not path.is_relative_to(root.resolve()):raise ValueError('Dataset path escapes its root')
    return path


def visible_files(root,pattern):
    for directory,folders,files in os.walk(root):
        folders[:]=[name for name in folders if name!='.lidar_studio_trash']
        for name in fnmatch.filter(files,pattern):yield Path(directory)/name


def samples(root):
    root=Path(root).expanduser().resolve()
    if not root.is_dir():raise ValueError('Choose a dataset folder')
    seen=set()
    for manifest in sorted(visible_files(root,'metadata.json')):
        if '.lidar_studio_trash' in manifest.parts:continue
        meta=json.loads(manifest.read_text())
        if meta.get('schema')!='lidar_studio.roi_dataset':continue
        if meta.get('status')!='complete':raise ValueError(f'Incomplete export: {manifest.parent}')
        for entry in meta['samples']:
            path=inside(manifest.parent,entry['points'])
            if path in seen:continue
            detail=json.loads(inside(manifest.parent,entry['metadata']).read_text())
            yield dict(path=path,root=manifest.parent,id=entry['id'],meta=detail,
                labels=manifest.parent/'labels'/(entry['id']+'.npy'),
                boxes=manifest.parent/'annotations'/(entry['id']+'.json'))
            seen.add(path)
    for path in sorted(visible_files(root,'coord.npy')):
        if '.lidar_studio_trash' in path.parts:continue
        if path in seen:continue
        metadata=path.parent/'meta.json'
        yield dict(path=path,root=path.parent,id=path.parent.name,
            meta=json.loads(metadata.read_text())if metadata.is_file()else{},
            labels=path.parent/'segment.npy',boxes=path.parent/'annotations.json')
        seen.add(path)
    if not seen:
        for path in sorted(visible_files(root,'*.npy')):
            if '.lidar_studio_trash' in path.parts:continue
            if path.parent.name in ('labels','metadata','annotations') or path.stem in ('segment','weak_segment','intensity','ring','point_time','point_index'):continue
            yield dict(path=path,root=root,id=path.stem,meta={},labels=root/'labels'/(path.stem+'.npy'),boxes=root/'annotations'/(path.stem+'.json'))


def points(sample):
    array=np.load(sample['path'],allow_pickle=False)
    if array.ndim!=2 or array.shape[1]not in (3,4)or array.dtype.kind!='f':raise ValueError(f'Expected floating N×3/4 points: {sample["path"]}')
    if len(array)==0 or len(array)>20_000_000 or not np.isfinite(array).all():raise ValueError(f'Empty, nonfinite or oversized point array: {sample["path"]}')
    if array.shape[1]==3:
        intensity=sample['path'].parent/'intensity.npy'
        strength=np.load(intensity,allow_pickle=False).reshape(-1)if intensity.is_file()else np.ones(len(array),dtype=np.float32)
        if len(strength)!=len(array)or not np.isfinite(strength).all():raise ValueError('Intensity count/values do not match coordinates')
        array=np.column_stack((array,strength))
    return array.astype('<f4',copy=False)


def group(sample):
    meta=sample['meta'];source=meta.get('source',{})
    # Entire recordings remain disjoint; split manifests may define narrower
    # sequences explicitly. Never randomly split adjacent crops from a scan.
    return str(meta.get('split_group')or source.get('path')or meta.get('source_bag')or sample['root'])


def split(records, fraction, seed):
    groups=sorted({group(s)for s in records})
    def frame(sample):
        meta=sample['meta'];source=meta.get('source',{})
        return int(source.get('frame_number')or meta.get('frame_index')or int(source.get('frame_index',-1))+1)
    temporal=len(groups)<2
    keys=[group(s)+':block:'+str(frame(s)//100)if temporal and frame(s)>0 else group(s)for s in records]
    groups=sorted(set(keys))
    if len(groups)<2:raise ValueError('Need independent recordings or at least two 100-frame time blocks for validation; provide a prepared split manifest otherwise')
    if not 0<fraction<.5:raise ValueError('Validation fraction must be between 0 and 0.5')
    order=np.random.default_rng(seed).permutation(len(groups));count=max(1,min(len(groups)-1,round(len(groups)*fraction)))
    validation={groups[i]for i in order[:count]}
    val=[i for i,key in enumerate(keys)if key in validation]
    train=[i for i,key in enumerate(keys)if key not in validation]
    if temporal:train=[i for i in train if all(abs(frame(records[i])-frame(records[j]))>10 for j in val)]
    if not train or not val:raise ValueError('No disjoint train/validation samples remain after the 10-frame boundary guard')
    return train,val


def labels(sample, count, allow_weak=False):
    path=sample['labels'];weak=False
    status=str(sample['meta'].get('annotation_status',''))+' '+str(sample['meta'].get('label_type',''))
    if any(word in status.lower()for word in('weak','pseudo','predicted')):
        if not allow_weak:raise ValueError('Weak or predicted labels require explicit opt-in')
        weak=True
    if path.is_file():values=np.load(path,allow_pickle=False).reshape(-1)
    else:values=np.full(count,-1,dtype=np.int64)
    if allow_weak and not (values>=0).any()and(path.parent/'weak_segment.npy').is_file():
        path=path.parent/'weak_segment.npy';values=np.load(path,allow_pickle=False).reshape(-1);weak=True
    if len(values)!=count or values.dtype.kind not in 'iu' or values.min(initial=-1)<-1 or values.max(initial=0)>2:raise ValueError(f'Expected N integer labels -1,0,1,2: {path}')
    if not(values>=0).any():raise ValueError(f'No point labels in {sample["id"]}; add labels/<id>.npy or select a prepared labeled split')
    return values.astype(np.int64),weak


def prepare_litept(root, output, fraction, seed, allow_weak):
    records=list(samples(root))
    if not records:raise ValueError('No NPY point samples found')
    train,val=split(records,fraction,seed);offsets=[0];weak=False
    for sample in records:
        array=np.load(sample['path'],mmap_mode='r',allow_pickle=False);offsets.append(offsets[-1]+len(array))
    output.mkdir(parents=True,exist_ok=True)
    coord=np.lib.format.open_memmap(output/'coord.npy',mode='w+',dtype='<f4',shape=(offsets[-1],3))
    intensity=np.lib.format.open_memmap(output/'intensity.npy',mode='w+',dtype='<f4',shape=(offsets[-1],))
    target_labels=np.lib.format.open_memmap(output/'labels.npy',mode='w+',dtype='<i8',shape=(offsets[-1],))
    known=set()
    for index,sample in enumerate(records):
        cloud=points(sample);target,is_weak=labels(sample,len(cloud),allow_weak);weak|=is_weak
        if index in train:known.update(np.unique(target).tolist())
        if sample['meta'].get('coordinate_space')=='aligned':
            transform=np.asarray(sample['meta'].get('sensor_to_output_row_major',[]),dtype=float)
            if transform.size!=16 or not np.isfinite(transform).all():raise ValueError('Aligned LitePT samples require their sensor-to-output transform')
            matrix=transform.reshape(4,4)
            if not np.allclose(matrix[3],[0,0,0,1])or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-4):raise ValueError('Expected a rigid sensor transform')
            cloud=cloud.copy();cloud[:,:3]=(cloud[:,:3]-matrix[:3,3])@matrix[:3,:3]
        begin,end=offsets[index:index+2];coord[begin:end]=cloud[:,:3];intensity[begin:end]=cloud[:,3];target_labels[begin:end]=target
    if 0 not in known or not({1,2}&known):raise ValueError('Segmentation training needs labeled background and foreground context; positive-only ROI crops are insufficient')
    cache=output/'contexts.npz';np.savez(cache,coord=coord,intensity=intensity,labels=target_labels,offsets=np.asarray(offsets,dtype=np.int64))
    del coord,intensity,target_labels
    for name in('coord.npy','intensity.npy','labels.npy'):(output/name).unlink()
    digest=hashlib.sha256()
    with cache.open('rb')as file:
        for block in iter(lambda:file.read(1024*1024),b''):digest.update(block)
    manifest=dict(context_npz=str(cache),train_indices=train,val_indices=val,test_indices=[],
        classes=['background','train_front','steel_structure'],label_type='weak ROI labels'if weak else'provided point annotations',
        split_strategy='recording groups or 100-frame blocks with 10-frame boundary guard',samples=[dict(path=str(s['path']),group=group(s))for s in records],
        training_npz_sha256=digest.hexdigest())
    path=output/'split.json';path.write_text(json.dumps(manifest,indent=2)+'\n');return path

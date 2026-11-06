"""MMDetection3D 1.x CenterPoint adapter for explicitly annotated Studio crops."""
import copy
import importlib.util
import json
from pathlib import Path
import pickle
import numpy as np
from .data import samples, points, split


def annotations(sample, classes):
    if not sample['boxes'].is_file():raise ValueError(f'Missing 3D boxes: {sample["boxes"]}. ROI selections and detector predictions are not ground truth')
    annotation=json.loads(sample['boxes'].read_text())
    if not isinstance(annotation,dict)or'boxes'not in annotation:raise ValueError('Annotation JSON needs a boxes array (empty is a negative example)')
    stored=sample['meta'].get('coordinate_space','sensor')
    if annotation.get('coordinate_space',stored)!=stored:raise ValueError('Annotation and cloud coordinate spaces differ')
    result=[]
    for box in annotation['boxes']:
        center=np.asarray(box['center_m'],dtype=float);size=np.asarray(box['size_m'],dtype=float);yaw=float(box['yaw_deg'])
        if center.shape!=(3,)or size.shape!=(3,)or not np.isfinite(np.r_[center,size,yaw]).all()or(size<=0).any():raise ValueError('3D box needs finite center_m/size_m/yaw_deg and positive sizes')
        if box['class_name']not in classes:raise ValueError(f'Class {box["class_name"]!r} is not in the selected class list')
        result.append(dict(bbox_3d=[*center,*size,np.deg2rad(yaw)],bbox_label_3d=classes.index(box['class_name'])))
    return result


def dataset_records(request):
    classes=[s.strip()for s in request.get('classes','person,car,bicycle,truck,bus').split(',')if s.strip()]
    if not classes or len(set(classes))!=len(classes):raise ValueError('Choose unique, nonempty training class names')
    records=list(samples(request['dataset']));train,val=split(records,request['validation_fraction'],request['seed'])
    spaces={s['meta'].get('coordinate_space','sensor')for s in records}
    if len(spaces)>1:raise ValueError('Choose samples in one coordinate space')
    instances=[annotations(s,classes)for s in records]
    if not any(instances[i]for i in train):raise ValueError('Training split contains no annotated objects')
    if not any(instances[i]for i in val):raise ValueError('Validation split needs annotated objects for box AP; add independent labeled samples')
    return classes,records,instances,train,val


def validate(request):
    if not Path(request.get('config','')).is_file():raise ValueError('Choose a CenterPoint base config (.py)')
    for package in ('torch','mmengine','mmdet3d'):
        if importlib.util.find_spec(package)is None:raise ValueError(f'Selected Python has no {package}; choose an MMDetection3D 1.x CUDA training environment')
    from mmengine.config import Config
    config=Config.fromfile(request['config'])
    if config.model.type!='CenterPoint':raise ValueError('This adapter supports a CenterPoint base config')
    import torch
    if not torch.cuda.is_available():raise ValueError('CenterPoint training requires a CUDA training environment')
    classes,records,_,train,val=dataset_records(request)
    for sample in records:points(sample)
    return dict(classes=classes,samples=len(records),train=len(train),validation=len(val),model='CenterPoint',framework='MMDetection3D 1.x')


def prepare(request, output, feature_count):
    classes,records,instances,train,val=dataset_records(request)
    output.mkdir(parents=True,exist_ok=True);(output/'points').mkdir()
    data=[]
    for index,sample in enumerate(records):
        cloud=points(sample)
        if feature_count==5:cloud=np.column_stack((cloud,np.zeros(len(cloud),dtype='<f4')))
        target=output/'points'/f'{index:08d}.bin';cloud.astype('<f4',copy=False).tofile(target)
        # Standard CenterHead requires velocity slots. Crops do not measure
        # velocity: their loss weights are zero, rather than teaching zero speed.
        padded=[dict(item,bbox_3d=[*item['bbox_3d'],0.,0.])for item in instances[index]]
        data.append(dict(sample_idx=index,lidar_points=dict(num_pts_feats=feature_count,lidar_path=str(target)),instances=padded))
    meta=dict(categories={name:i for i,name in enumerate(classes)},dataset='StudioDataset',info_version='1.0')
    for name,indices in [('train',train),('val',val)]:
        with(output/f'infos_{name}.pkl').open('wb')as file:pickle.dump(dict(metainfo=meta,data_list=[data[i]for i in indices]),file)
    return classes


def configure_head(model, classes):
    head=model['pts_bbox_head'];old=head['tasks'];mapping=[]
    for name in classes:
        alias='pedestrian'if name=='person'else name
        mapping.append(next(((i,t['class_names'].index(alias))for i,t in enumerate(old)if alias in t['class_names']),(0,0)))
    head['tasks']=[dict(num_class=1,class_names=[name])for name in classes]
    if'vel'not in head.get('common_heads',{}):raise ValueError('The base CenterHead must retain its velocity output slots')
    for config in(head.get('train_cfg',{}),model.get('train_cfg',{}).get('pts',{})):
        weights=list(config.get('code_weights',[1.]*10))
        if len(weights)!=10:raise ValueError('Expected standard 10-channel CenterHead regression weights')
        config['code_weights']=[*weights[:8],0.,0.]
    for config in(head.get('test_cfg',{}),model.get('test_cfg',{}).get('pts',{})):
        for key in('min_radius','nms_type','nms_thr','nms_rescale_factor'):
            values=config.get(key)
            if isinstance(values,(list,tuple))and len(values)==len(old):
                selected=[]
                for task,category in mapping:
                    value=values[task]
                    selected.append([value[min(category,len(value)-1)]]if isinstance(value,(list,tuple))else value)
                config[key]=selected


def configure_schedule(config, request):
    previous_epochs=int(config.train_cfg['max_epochs']);previous_lr=float(config.optim_wrapper['optimizer']['lr'])
    configured=[]
    for item in copy.deepcopy(config.get('param_scheduler',[])):
        if not item.get('by_epoch',True):raise ValueError('Use an epoch-based CenterPoint schedule')
        begin=round(item.get('begin',0)*request['epochs']/previous_epochs)
        end=round(item.get('end',previous_epochs)*request['epochs']/previous_epochs)
        end=min(request['epochs'],end)
        if end<=begin:continue
        item.update(begin=begin,end=end)
        if 'T_max'in item:item['T_max']=end-begin
        if 'milestones'in item:item['milestones']=sorted({round(v*request['epochs']/previous_epochs)for v in item['milestones']})
        if 'LR'in item['type']and'eta_min'in item:item['eta_min']*=request['learning_rate']/previous_lr
        configured.append(item)
    config.param_scheduler=configured
    config.optim_wrapper['optimizer']['lr']=request['learning_rate']
    config.train_cfg.update(max_epochs=request['epochs'],val_interval=1)
    config.auto_scale_lr=dict(enable=False)


def run(request, output):
    from mmengine.config import Config
    from mmengine.runner import Runner
    from mmdet3d.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    from .mmdet_dataset import StudioDataset
    config=Config.fromfile(request['config'])
    feature_count=int(config.model.get('pts_voxel_encoder',{}).get('in_channels',4))
    if feature_count not in(4,5):raise ValueError('CenterPoint input must use 4 or 5 point features')
    classes=prepare(request,output/'prepared',feature_count)
    # The dataset's class list belongs to this run, not to the source model.
    StudioDataset.METAINFO={'classes':tuple(classes),'box_type_3d':'LiDAR'}
    configure_head(config.model,classes)
    def pipeline(original):
        result=[]
        for item in copy.deepcopy(original):
            kind=item.get('type','')
            if kind in('ObjectSample','LoadPointsFromMultiSweeps'):continue
            if kind=='LoadPointsFromFile':item.update(load_dim=feature_count,use_dim=list(range(feature_count)))
            if kind=='ObjectNameFilter':item['classes']=classes
            if'transforms'in item:item['transforms']=pipeline(item['transforms'])
            result.append(item)
        return result
    def loader(original, training):
        inner=original['dataset']
        while'dataset'in inner:inner=inner['dataset']
        return dict(batch_size=request['batch_size']if training else 1,num_workers=request['workers'],
            persistent_workers=request['workers']>0,sampler=dict(type='DefaultSampler',shuffle=training),
            dataset=dict(type='StudioDataset',data_root=str(output/'prepared'),data_prefix=dict(pts=''),
                ann_file=f'infos_{"train"if training else"val"}.pkl',pipeline=pipeline(inner['pipeline']),
                metainfo=dict(classes=classes),modality=dict(use_lidar=True,use_camera=False),box_type_3d='LiDAR',
                filter_empty_gt=False,test_mode=not training))
    config.train_dataloader=loader(config.train_dataloader,True);config.val_dataloader=loader(config.val_dataloader,False)
    config.val_evaluator=dict(type='StudioBoxMetric',iou_thr=[.25,.5],prefix='box3d')
    config.test_dataloader=copy.deepcopy(config.val_dataloader)
    config.test_evaluator=copy.deepcopy(config.val_evaluator)
    configure_schedule(config,request)
    if'checkpoint'in config.get('default_hooks',{}):config.default_hooks.checkpoint.update(save_best='box3d/mAP_0.50',rule='greater')
    config.randomness=dict(seed=request['seed'])
    config.work_dir=str(output/'model');config.load_from=None;config.resume=False;config.default_scope='mmdet3d'
    config.dump(str(output/'resolved_config.py'))
    print('Training CenterPoint; generic 3D box AP at IoU 0.25/0.5; original source config is preserved',flush=True)
    from mmengine.hooks import Hook
    class Progress(Hook):
        def after_train_epoch(self, runner):
            row=dict(epoch=runner.epoch+1)
            try:row['train_loss']=float(runner.message_hub.get_scalar('train/loss').current())
            except(KeyError,ValueError):pass
            pending=output/'progress.tmp';pending.write_text(json.dumps(row));pending.replace(output/'progress.json')
    runner=Runner.from_cfg(config);runner.register_hook(Progress(),priority='LOW');runner.train()

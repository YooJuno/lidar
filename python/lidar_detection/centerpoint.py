# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 TIER IV, Inc. (original pre/post-processing conventions)
# Pre/post-processing follows TIER IV's Autoware lidar_centerpoint implementation.
# Source: autowarefoundation/autoware_universe @ 7645a923be5357ad9480b012db52da4c4940f52d
"""Original Autoware CenterPoint tiny weights, portable ONNX inference.

Input is one full, gravity-aligned XYZ scan in vehicle/ground coordinates.
The fourth model feature is time lag (zero for a single scan), NOT intensity.
"""
from pathlib import Path
import hashlib
import math
import time
import numpy as np

MODEL_REVISION='494c8171def40bd36cc2feb323e0a5acbfab132b'
MODEL_FILES={
    'pts_voxel_encoder.onnx':'2c53465715c1fd2e9dc5727ef3fca74f4cdf0538f74286b0946e219d0ca5693b',
    'pts_backbone_neck_head.onnx':'9bb0b634f3664bd098ce7d6a3d8a9fb7cc8d9b8252b27f302c71e43316bab551',
}
CLASS_NAMES=('Car','Truck','Bus','Bicycle','Person')
RANGE_MIN=np.array([-76.8,-76.8,-4.],dtype=np.float32)
RANGE_MAX=np.array([76.8,76.8,6.],dtype=np.float32)
VOXEL_SIZE=np.float32(.32)
GRID_SIZE=480
MAX_VOXELS=40000
MAX_POINTS=32
FUSED_NAME='centerpoint-tiny-v4.1-portable-v1.onnx'


def verify_weights(folder):
    for name,expected in MODEL_FILES.items():
        path=Path(folder)/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Missing or invalid model file: {name}. Run detector setup again.')


def fuse_model(folder):
    """Join original PFN/head with the exact zero-fill + scatter used by Autoware."""
    import onnx
    from onnx import helper,numpy_helper,TensorProto,compose
    folder=Path(folder);verify_weights(folder)
    encoder=compose.add_prefix(onnx.load(folder/'pts_voxel_encoder.onnx'),'vfe/')
    head=compose.add_prefix(onnx.load(folder/'pts_backbone_neck_head.onnx'),'head/')
    indices=helper.make_tensor_value_info('voxel_indices',TensorProto.INT64,['num_voxels',3])
    shape=numpy_helper.from_array(np.array([1,GRID_SIZE,GRID_SIZE,32],dtype=np.int64),'bev_shape')
    bridge=[
        helper.make_node('Squeeze',[encoder.graph.output[0].name],['pillar_vectors'],axes=[1]),
        helper.make_node('ConstantOfShape',['bev_shape'],['empty_bev'],value=helper.make_tensor('zero',TensorProto.FLOAT,[1],[0.])),
        helper.make_node('ScatterND',['empty_bev','voxel_indices','pillar_vectors'],['bev_nhwc']),
        helper.make_node('Transpose',['bev_nhwc'],[head.graph.input[0].name],perm=[0,3,1,2]),
    ]
    graph=helper.make_graph([*encoder.graph.node,*bridge,*head.graph.node],
        'Autoware-CenterPoint-tiny-portable',[*encoder.graph.input,indices],head.graph.output,
        [*encoder.graph.initializer,shape,*head.graph.initializer])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',11)],producer_name='LiDAR-Studio-Autoware-adapter')
    model.ir_version=6
    helper.set_model_props(model,{'source_repository':'AutowareFoundation/lidar_centerpoint',
        'source_revision':MODEL_REVISION,'license':'Apache-2.0','features':'XYZT, time=0 for single scan',
        'adapter':'portable-v1; original neural network weights unchanged'})
    onnx.checker.check_model(model)
    target=folder/FUSED_NAME;temporary=target.with_suffix('.tmp');onnx.save(model,temporary);temporary.replace(target)
    return target


def pillarize(points):
    """32 points/pillar, XYZ+time+cluster offsets+XY center offsets (9 features)."""
    points=np.asarray(points,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError('Expected Nx3 XYZ points')
    valid=np.isfinite(points).all(axis=1)&(points>=RANGE_MIN).all(axis=1)&(points<RANGE_MAX).all(axis=1)
    xyz=points[valid]
    if not len(xyz):return np.zeros((0,32,9),np.float32),np.zeros((0,3),np.int64),{'pillars':0,'truncated_pillars':0,'points_in_range':0}
    # Autoware shuffles before its bounded voxelization. A fixed permutation
    # avoids scan-order bias while keeping a paused frame reproducible.
    xyz=xyz[np.random.default_rng(0).permutation(len(xyz))]
    xy=np.clip(np.floor((xyz[:,:2]-RANGE_MIN[:2])/VOXEL_SIZE).astype(np.int32),0,GRID_SIZE-1)
    keys=xy[:,1]*GRID_SIZE+xy[:,0]
    order=np.argsort(keys,kind='stable');xyz=xyz[order];xy=xy[order];keys=keys[order]
    starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1]
    counts=np.diff(np.r_[starts,len(keys)])
    total_pillars=len(starts)
    # Near pillars get priority only when the documented 40000-pillar cap is
    # exceeded; report the truncation rather than hiding it.
    if total_pillars>MAX_VOXELS:
        centers=(xy[starts].astype(np.float32)+.5)*VOXEL_SIZE+RANGE_MIN[:2]
        keep=np.zeros(total_pillars,dtype=bool)
        keep[np.argsort((centers*centers).sum(axis=1),kind='stable')[:MAX_VOXELS]]=True
    else:keep=np.ones(total_pillars,dtype=bool)
    group=np.repeat(np.arange(total_pillars),counts)
    rank=np.arange(len(keys))-np.repeat(starts,counts)
    pick=(rank<MAX_POINTS)&keep[group]
    remap=np.cumsum(keep)-1;ids=remap[group[pick]];rank=rank[pick];selected=xyz[pick]
    num=int(keep.sum());features=np.zeros((num,MAX_POINTS,9),dtype=np.float32)
    num_points=np.minimum(counts[keep],MAX_POINTS).astype(np.float32)
    means=np.stack([np.bincount(ids,weights=selected[:,axis],minlength=num) for axis in range(3)],axis=1).astype(np.float32)
    means/=num_points[:,None]
    voxel_xy=xy[starts][keep]
    centers=(voxel_xy.astype(np.float32)+.5)*VOXEL_SIZE+RANGE_MIN[:2]
    features[ids,rank,:3]=selected
    features[ids,rank,4:7]=selected-means[ids]
    features[ids,rank,7:9]=selected[:,:2]-centers[ids]
    indices=np.column_stack([np.zeros(num,dtype=np.int64),voxel_xy[:,1],voxel_xy[:,0]]).astype(np.int64)
    return features,indices,{'pillars':num,'truncated_pillars':total_pillars-num,'points_in_range':len(xyz)}


def box_corners(box):
    # box: x,y,z,length,width,height,ROS yaw,score,class
    local=np.array([[-.5,-.5],[.5,-.5],[.5,.5],[-.5,.5]],dtype=np.float64)*box[3:5]
    c,s=math.cos(float(box[6])),math.sin(float(box[6]))
    return local@np.array([[c,s],[-s,c]])+box[:2]


def polygon_area(polygon):
    if len(polygon)<3:return 0.
    p=np.asarray(polygon)
    return abs(float(np.dot(p[:,0],np.roll(p[:,1],1))-np.dot(p[:,1],np.roll(p[:,0],1))))*.5


def rotated_iou(a,b,corners_a=None,corners_b=None):
    polygon=list(box_corners(a) if corners_a is None else corners_a)
    clip=box_corners(b) if corners_b is None else corners_b
    for i in range(4):
        edge_a,edge_b=clip[i],clip[(i+1)%4];edge=edge_b-edge_a
        def side(p):return float(edge[0]*(p[1]-edge_a[1])-edge[1]*(p[0]-edge_a[0]))
        if not polygon:return 0.
        incoming=polygon;polygon=[];previous=incoming[-1];previous_side=side(previous)
        for current in incoming:
            current_side=side(current)
            if (current_side>=0)!=(previous_side>=0):
                t=previous_side/(previous_side-current_side)
                polygon.append(previous+(current-previous)*t)
            if current_side>=0:polygon.append(current)
            previous,previous_side=current,current_side
    intersection=polygon_area(polygon)
    return intersection/max(float(a[3]*a[4]+b[3]*b[4])-intersection,1e-8)


def suppress(boxes):
    if not len(boxes):return boxes
    # Original 0.5 m circle NMS, followed by rotated BEV IoU NMS (0.1).
    boxes=boxes[np.argsort(-boxes[:,7],kind='stable')]
    kept=[];corners=[];aabbs=[]
    for box in boxes:
        corner=box_corners(box);aabb=np.r_[corner.min(axis=0),corner.max(axis=0)]
        rejected=False
        for previous,previous_corners,previous_aabb in zip(kept,corners,aabbs):
            distance=float(np.square(box[:2]-previous[:2]).sum())
            if distance<.25:rejected=True;break
            if distance>=100:continue
            overlap=np.minimum(aabb[2:],previous_aabb[2:])-np.maximum(aabb[:2],previous_aabb[:2])
            if np.any(overlap<=0):continue
            if rotated_iou(box,previous,corner,previous_corners)>.1:rejected=True;break
        if not rejected:kept.append(box);corners.append(corner);aabbs.append(aabb)
    return np.asarray(kept,dtype=np.float32).reshape(-1,9)


def decode(outputs,score_threshold=.35):
    values={name.split('/')[-1]:value for name,value in outputs.items()}
    heatmap=values['heatmap'][0]
    labels=np.argmax(heatmap,axis=0)
    logits=np.max(heatmap,axis=0)
    scores=1/(1+np.exp(-np.clip(logits,-80,80)))
    ys,xs=np.nonzero(scores>=score_threshold)
    if not len(xs):return np.empty((0,9),np.float32)
    labels=labels[ys,xs];scores=scores[ys,xs]
    reg=values['reg'][0,:,ys,xs];height=values['height'][0,0,ys,xs]
    dimensions=values['dim'][0,:,ys,xs]
    rotation=values['rot'][0,:,ys,xs]
    norm=np.sqrt(np.square(rotation).sum(axis=1))
    keep=(norm>=np.array([.3,.3,.3,.3,0],np.float32)[labels])
    xs,ys,labels,scores,reg,height,dimensions,rotation=(v[keep] for v in [xs,ys,labels,scores,reg,height,dimensions,rotation])
    boxes=np.column_stack([(xs+reg[:,0])*.64+RANGE_MIN[0],(ys+reg[:,1])*.64+RANGE_MIN[1],height,
        np.exp(np.clip(dimensions[:,1],-6,6)),np.exp(np.clip(dimensions[:,0],-6,6)),np.exp(np.clip(dimensions[:,2],-6,6)),
        -np.arctan2(rotation[:,0],rotation[:,1])-np.pi/2,scores,labels]).astype(np.float32)
    boxes=boxes[np.isfinite(boxes).all(axis=1)]
    return suppress(boxes)


class CenterPoint:
    def __init__(self,folder,provider='auto'):
        import onnxruntime as ort
        self.folder=Path(folder);verify_weights(self.folder)
        path=self.folder/FUSED_NAME
        if not path.is_file():fuse_model(self.folder)
        # Use the host's existing PyTorch CUDA/cuDNN runtime when available.
        # No custom CUDA compiler, TensorRT plugin or ROS node is required.
        if provider!='cpu':
            try:import torch  # noqa: F401
            except ImportError:pass
            try:ort.preload_dlls()
            except AttributeError:pass
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        providers=['CPUExecutionProvider']
        if provider!='cpu' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers=[('CUDAExecutionProvider',{'cudnn_conv_algo_search':'HEURISTIC','arena_extend_strategy':'kSameAsRequested'}),*providers]
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
        self.provider=self.session.get_providers()[0]
        self.output_names=[value.name for value in self.session.get_outputs()]
        warm_features=np.zeros((1,32,9),np.float32);warm_indices=np.zeros((1,3),np.int64)
        self.session.run(None,{'vfe/input_features':warm_features,'voxel_indices':warm_indices})

    def infer(self,points,score_threshold=.35):
        started=time.perf_counter();features,indices,stats=pillarize(points)
        prepared=time.perf_counter()
        if not len(indices):return np.empty((0,9),np.float32),{**stats,'latency_ms':(prepared-started)*1000,'provider':self.provider}
        values=self.session.run(None,{'vfe/input_features':features,'voxel_indices':indices})
        inferred=time.perf_counter();boxes=decode(dict(zip(self.output_names,values)),score_threshold)
        stats.update(preprocess_ms=(prepared-started)*1000,inference_ms=(inferred-prepared)*1000,
            postprocess_ms=(time.perf_counter()-inferred)*1000,latency_ms=(time.perf_counter()-started)*1000,provider=self.provider)
        return boxes,stats

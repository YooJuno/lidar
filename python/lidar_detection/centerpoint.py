# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 TIER IV, Inc. (original pre/post-processing conventions)
# Pre/post-processing follows TIER IV's Autoware lidar_centerpoint implementation.
# Source: autowarefoundation/autoware_universe @ 7645a923be5357ad9480b012db52da4c4940f52d
"""Original TIER IV CenterPoint weights, portable ONNX inference.

Input is one full, gravity-aligned XYZ scan in vehicle/ground coordinates.
The fourth model feature is time lag (zero for a single scan), NOT intensity.
"""
from pathlib import Path
import hashlib
import math
import time
import numpy as np
from .native import stable_order,fill_features,backend_name
from .models import get_model,AWML_REVISION

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
FUSED_NAME=get_model('tiny').fused_name


def verify_weights(folder,model='tiny'):
    for name,_,expected in get_model(model).files:
        if not name.endswith('.onnx'):continue
        path=Path(folder)/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Missing or invalid model file: {name}. Run detector setup again.')


def fuse_model(folder,model='tiny'):
    """Join original PFN/head with the exact zero-fill + scatter used by Autoware."""
    import onnx
    from onnx import helper,numpy_helper,TensorProto,compose
    spec=get_model(model);folder=Path(folder);verify_weights(folder,model)
    encoder=onnx.load(folder/'pts_voxel_encoder.onnx');head=onnx.load(folder/'pts_backbone_neck_head.onnx')
    opset=max(o.version for m in [encoder,head] for o in m.opset_import if o.domain=='')
    from onnx import version_converter
    encoder=compose.add_prefix(version_converter.convert_version(encoder,opset),'vfe/')
    head=compose.add_prefix(version_converter.convert_version(head,opset),'head/')
    indices=helper.make_tensor_value_info('voxel_indices',TensorProto.INT64,['num_voxels',3])
    shape=numpy_helper.from_array(np.array([1,spec.grid,spec.grid,32],dtype=np.int64),'bev_shape')
    axes=numpy_helper.from_array(np.array([1],np.int64),'squeeze_axes')
    squeeze=helper.make_node('Squeeze',[encoder.graph.output[0].name,'squeeze_axes'],['pillar_vectors']) if opset>=13 else \
        helper.make_node('Squeeze',[encoder.graph.output[0].name],['pillar_vectors'],axes=[1])
    bridge=[squeeze,
        helper.make_node('ConstantOfShape',['bev_shape'],['empty_bev'],value=helper.make_tensor('zero',TensorProto.FLOAT,[1],[0.])),
        helper.make_node('ScatterND',['empty_bev','voxel_indices','pillar_vectors'],['bev_nhwc']),
        helper.make_node('Transpose',['bev_nhwc'],[head.graph.input[0].name],perm=[0,3,1,2]),
    ]
    graph=helper.make_graph([*encoder.graph.node,*bridge,*head.graph.node],
        'TIER-IV-CenterPoint-portable',[*encoder.graph.input,indices],head.graph.output,
        [*encoder.graph.initializer,shape,*([axes] if opset>=13 else []),*head.graph.initializer])
    fused=helper.make_model(graph,opset_imports=[helper.make_opsetid('',opset)],producer_name='LiDAR-Studio-Autoware-adapter')
    fused.ir_version=8
    helper.set_model_props(fused,{'source_repository':'tier4/AWML' if model=='short_range' else 'AutowareFoundation/lidar_centerpoint',
        'source_revision':AWML_REVISION if model=='short_range' else MODEL_REVISION,'license':'Apache-2.0','features':'XYZT, time=0 for single scan',
        'adapter':'portable-v2; original neural network weights unchanged','variant':spec.name})
    onnx.checker.check_model(fused)
    target=folder/spec.fused_name;temporary=target.with_suffix('.tmp');onnx.save(fused,temporary);temporary.replace(target)
    return target


def pillarize(points,model='tiny'):
    """Deterministic 32-point pillars with the selected model's original features."""
    spec=get_model(model)
    minimum=np.array(spec.minimum,np.float32);maximum=np.array(spec.maximum,np.float32)
    voxel=np.float32(spec.voxel[0])
    points=np.asarray(points,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=3:raise ValueError('Expected Nx3 XYZ points')
    x,y,z=points[:,0],points[:,1],points[:,2]
    valid=(x>=minimum[0])&(x<maximum[0])&(y>=minimum[1])&(y<maximum[1])&(z>=minimum[2])&(z<maximum[2])
    xyz=points[valid]
    if not len(xyz):return np.zeros((0,32,spec.features),np.float32),np.zeros((0,3),np.int64),{'pillars':0,'truncated_pillars':0,'points_in_range':0,'preprocess_backend':backend_name()}
    x=np.clip(np.floor((xyz[:,0]-minimum[0])/voxel).astype(np.int32),0,spec.grid-1)
    y=np.clip(np.floor((xyz[:,1]-minimum[1])/voxel).astype(np.int32),0,spec.grid-1)
    keys=y*spec.grid+x
    permutation=np.random.default_rng(0).permutation(len(xyz))
    # Sort indices, not XYZ/XY copies. Counting sort preserves the original
    # shuffle exactly; both native and NumPy paths keep identical samples.
    order=stable_order(keys,permutation,spec.grid)
    keys=keys[order]
    starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1]
    counts=np.diff(np.r_[starts,len(keys)])
    voxel_keys=keys[starts];voxel_xy=np.column_stack([voxel_keys%spec.grid,voxel_keys//spec.grid])
    total_pillars=len(starts)
    if total_pillars>spec.max_voxels:
        centers=(voxel_xy.astype(np.float32)+.5)*voxel+minimum[:2]
        keep=np.zeros(total_pillars,dtype=bool)
        keep[np.argsort((centers*centers).sum(axis=1),kind='stable')[:spec.max_voxels]]=True
    else:keep=np.ones(total_pillars,dtype=bool)
    num=int(keep.sum());features=np.zeros((num,MAX_POINTS,spec.features),dtype=np.float32)
    if not fill_features(xyz,order,starts[keep],counts[keep],voxel_keys[keep],features,spec.grid,spec.voxel[0],spec.minimum,spec.voxel[2]):
        group=np.repeat(np.arange(total_pillars),counts)
        rank=np.arange(len(keys))-np.repeat(starts,counts)
        pick=(rank<MAX_POINTS)&keep[group]
        remap=np.cumsum(keep)-1;ids=remap[group[pick]];rank=rank[pick];selected=xyz[order[pick]]
        num_points=np.minimum(counts[keep],MAX_POINTS).astype(np.float32)
        means=np.stack([np.bincount(ids,weights=selected[:,axis],minlength=num) for axis in range(3)],axis=1).astype(np.float32)
        means/=num_points[:,None]
        selected_xy=voxel_xy[keep]
        centers=(selected_xy.astype(np.float32)+.5)*voxel+minimum[:2]
        features[ids,rank,:3]=selected
        features[ids,rank,4:7]=selected-means[ids]
        features[ids,rank,7:9]=selected[:,:2]-centers[ids]
        if spec.features==10:features[ids,rank,9]=selected[:,2]-np.float32(spec.minimum[2]+spec.voxel[2]*.5)
    voxel_xy=voxel_xy[keep]
    indices=np.column_stack([np.zeros(num,dtype=np.int64),voxel_xy[:,1],voxel_xy[:,0]]).astype(np.int64)
    return features,indices,{'pillars':num,'truncated_pillars':total_pillars-num,'points_in_range':len(xyz),'preprocess_backend':backend_name(len(xyz))}


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
    # Four/eight vertices: scalar double arithmetic avoids allocating tiny
    # NumPy arrays for every edge intersection. Keep the same area reduction.
    polygon=[(float(p[0]),float(p[1])) for p in (box_corners(a) if corners_a is None else corners_a)]
    clip=box_corners(b) if corners_b is None else corners_b
    for i in range(4):
        ax,ay=float(clip[i,0]),float(clip[i,1]);bx,by=float(clip[(i+1)%4,0]),float(clip[(i+1)%4,1]);ex,ey=bx-ax,by-ay
        if not polygon:return 0.
        incoming=polygon;polygon=[];px,py=incoming[-1];previous_side=ex*(py-ay)-ey*(px-ax)
        for cx,cy in incoming:
            side=ex*(cy-ay)-ey*(cx-ax)
            if (side>=0)!=(previous_side>=0):
                t=previous_side/(previous_side-side)
                polygon.append((px+(cx-px)*t,py+(cy-py)*t))
            if side>=0:polygon.append((cx,cy))
            px,py,previous_side=cx,cy,side
    intersection=polygon_area(polygon)
    return intersection/max(float(a[3]*a[4]+b[3]*b[4])-intersection,1e-8)

def suppress(boxes):
    if not len(boxes):return boxes
    boxes=boxes[np.argsort(-boxes[:,7],kind='stable')]
    kept=[];corners=[];aabbs=[];cells={}
    for box in boxes:
        corner=box_corners(box);lo=corner.min(axis=0);hi=corner.max(axis=0)
        aabb=(float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1]))
        rejected=False
        # Original 0.5 m circle and 0.1 IoU suppression, within 10 m only.
        cell=(math.floor(float(box[0])/10),math.floor(float(box[1])/10))
        neighbors=(index for y in range(cell[1]-1,cell[1]+2) for x in range(cell[0]-1,cell[0]+2) for index in cells.get((x,y),()))
        for index in neighbors:
            previous,previous_corners,p=kept[index],corners[index],aabbs[index]
            dx=box[0]-previous[0];dy=box[1]-previous[1];distance=float(dx*dx+dy*dy)
            if distance<.25:rejected=True;break
            if distance>=100:continue
            if min(aabb[2],p[2])<=max(aabb[0],p[0]) or min(aabb[3],p[3])<=max(aabb[1],p[1]):continue
            if rotated_iou(box,previous,corner,previous_corners)>.1:rejected=True;break
        if not rejected:
            cells.setdefault(cell,[]).append(len(kept));kept.append(box);corners.append(corner);aabbs.append(aabb)
    return np.asarray(kept,dtype=np.float32).reshape(-1,9)


def decode(outputs,score_threshold=.35,model='tiny'):
    spec=get_model(model)
    values={name.split('/')[-1]:value for name,value in outputs.items()}
    heatmap=values['heatmap'][0]
    if heatmap.shape!=(5,spec.grid//spec.downsample,spec.grid//spec.downsample):raise ValueError('Model head geometry does not match the selected model')
    step=spec.voxel[0]*spec.downsample;minimum=spec.minimum
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
    boxes=np.column_stack([(xs+reg[:,0])*step+minimum[0],(ys+reg[:,1])*step+minimum[1],height,
        np.exp(np.clip(dimensions[:,1],-6,6)),np.exp(np.clip(dimensions[:,0],-6,6)),np.exp(np.clip(dimensions[:,2],-6,6)),
        -np.arctan2(rotation[:,0],rotation[:,1])-np.pi/2,scores,labels]).astype(np.float32)
    boxes=boxes[np.isfinite(boxes).all(axis=1)]
    return suppress(boxes)


class CenterPoint:
    def __init__(self,folder,provider='auto',model='tiny',*,use_tf32=True):
        import onnxruntime as ort
        self.spec=get_model(model);self.model=model
        self.folder=Path(folder);verify_weights(self.folder,model)
        path=self.folder/self.spec.fused_name
        if not path.is_file():fuse_model(self.folder,model)
        # ORT can preload the NVIDIA wheels directly. Importing all of PyTorch
        # only to load CUDA libraries wastes startup time and resident memory.
        available=ort.get_available_providers()
        if provider!='cpu' and 'CUDAExecutionProvider' in available:
            # With system-site-packages, NVIDIA wheels may live outside ORT's
            # own venv. Resolve the actual cuDNN package before ORT's default
            # relative-directory search; do not import the training framework.
            import importlib.util
            import platform
            try:
                spec=importlib.util.find_spec('nvidia.cudnn')
            except ImportError:spec=None
            try:
                locations=(spec.submodule_search_locations or []) if spec else []
                for root in locations:
                    directory=Path(root)/('bin' if platform.system()=='Windows' else 'lib')
                    library='cudnn64_9.dll' if platform.system()=='Windows' else 'libcudnn.so.9'
                    if (directory/library).is_file():
                        ort.preload_dlls(cuda=False,cudnn=True,directory=str(directory));break
                ort.preload_dlls()
            except AttributeError:pass
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        providers=['CPUExecutionProvider']
        if provider!='cpu' and 'CUDAExecutionProvider' in available:
            # ConvNeXt's depthwise convolutions benefit from ORT's CUDA NHWC
            # layout. Keep the strict numerical comparison on plain FP32 NCHW.
            providers=[('CUDAExecutionProvider',{'cudnn_conv_algo_search':'HEURISTIC',
                'arena_extend_strategy':'kSameAsRequested','use_tf32':int(use_tf32),
                'prefer_nhwc':int(model=='short_range' and use_tf32)}),*providers]
        self.session=ort.InferenceSession(str(path),sess_options=options,providers=providers)
        self.provider=self.session.get_providers()[0]
        self.output_names=[value.name for value in self.session.get_outputs()]
        warm_features=np.zeros((1,32,self.spec.features),np.float32);warm_indices=np.zeros((1,3),np.int64)
        self.session.run(None,{'vfe/input_features':warm_features,'voxel_indices':warm_indices})

    def infer(self,points,score_threshold=.35):
        started=time.perf_counter();features,indices,stats=pillarize(points,self.model)
        prepared=time.perf_counter()
        if not len(indices):return np.empty((0,9),np.float32),{**stats,'latency_ms':(prepared-started)*1000,'provider':self.provider}
        values=self.session.run(None,{'vfe/input_features':features,'voxel_indices':indices})
        inferred=time.perf_counter();boxes=decode(dict(zip(self.output_names,values)),score_threshold,self.model)
        stats.update(preprocess_ms=(prepared-started)*1000,inference_ms=(inferred-prepared)*1000,
            postprocess_ms=(time.perf_counter()-inferred)*1000,latency_ms=(time.perf_counter()-started)*1000,provider=self.provider)
        return boxes,stats

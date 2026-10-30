# SPDX-License-Identifier: Apache-2.0
# Frozen NumPy oracle from 18f478e (before native preprocessing). Kept outside
# the runtime to independently verify sampling, float32 rounding and clipping.
import numpy as np
import math
from lidar_detection.centerpoint import RANGE_MIN,RANGE_MAX,VOXEL_SIZE,GRID_SIZE,MAX_VOXELS,MAX_POINTS

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

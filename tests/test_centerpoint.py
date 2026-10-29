#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fixtures for original CenterPoint coordinates, features and NMS.
Optional --model-dir checks fused-vs-original neural graph numerically on a
real cloud (--cloud KITTI_XYZI.bin), then reports actual inference timings.
"""
import argparse
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from lidar_detection.centerpoint import pillarize,decode,rotated_iou,suppress,CenterPoint

class PrePostTests(unittest.TestCase):
    def test_pillar_features_and_padding(self):
        xyz=np.array([[1,2,3],[1.1,2.1,3.2],[np.nan,0,0],[80,0,0]],np.float32)
        features,indices,stats=pillarize(xyz)
        self.assertEqual(stats['points_in_range'],2);self.assertEqual(len(indices),1)
        self.assertEqual(indices.tolist(),[[0,246,243]])
        observed=features[0,:2];np.testing.assert_allclose(observed[:,:3].mean(axis=0),[1.05,2.05,3.1],atol=1e-6)
        self.assertTrue(np.all(features[:,:,3]==0)) # time lag, never intensity
        np.testing.assert_allclose(observed[:,4:7].sum(axis=0),0,atol=1e-6)
        center=np.array([-76.8+(243+.5)*.32,-76.8+(246+.5)*.32])
        np.testing.assert_allclose(observed[:,7:9],observed[:,:2]-center,atol=1e-5)
        self.assertTrue(np.all(features[0,2:]==0))
        f,i,_=pillarize(xyz);np.testing.assert_array_equal(f,features);np.testing.assert_array_equal(i,indices)
    def test_limits_and_cap(self):
        p=np.repeat([[0.,0.,0.]],80,axis=0).astype(np.float32)
        f,i,s=pillarize(p);self.assertEqual(f.shape,(1,32,9));self.assertEqual(s['pillars'],1)
        p=np.array([[-76.8,0,-4],[76.8,0,0],[0,76.8,0],[0,0,6],[0,0,np.inf]],np.float32)
        _,_,s=pillarize(p);self.assertEqual(s['points_in_range'],1)
    def test_decoder_ros_axes_and_dimensions(self):
        arrays={name:np.zeros((1,ch,240,240),np.float32) for name,ch in [('heatmap',5),('reg',2),('height',1),('dim',3),('rot',2)]}
        arrays['heatmap'][:]=-20;arrays['heatmap'][0,4,100,120]=2
        arrays['reg'][0,:,100,120]=[.25,.5];arrays['height'][0,0,100,120]=.9
        arrays['dim'][0,:,100,120]=np.log([.6,1.2,1.8]);arrays['rot'][0,:,100,120]=[0,1]
        box=decode(arrays,.35)[0]
        # Expected independently from model manifest + Autoware ROS conversion.
        np.testing.assert_allclose(box[:7],[.16,-12.48,.9,1.2,.6,1.8,-np.pi/2],atol=1e-5)
        self.assertEqual(int(box[8]),4);self.assertAlmostEqual(float(box[7]),1/(1+np.exp(-2)),places=6)
        arrays['heatmap'][0,4,100,120]=-20
        self.assertEqual(decode(arrays,.35).shape,(0,9))
    def test_rotated_nms(self):
        a=np.array([0,0,1,4,2,2,0,.9,0],np.float32);b=a.copy();b[6]=np.pi/2
        self.assertAlmostEqual(rotated_iou(a,a),1,places=6)
        self.assertAlmostEqual(rotated_iou(a,b),1/3,places=6)
        far=a.copy();far[0]=20
        self.assertEqual(len(suppress(np.stack([a,b,far]))),2)
        low=a.copy();low[0]=10;low[7]=.2
        candidates=np.stack([a,b,far,low]);cached=suppress(candidates)
        np.testing.assert_array_equal(cached[cached[:,7]>=.35],suppress(candidates[candidates[:,7]>=.35]))

def model_check(folder,cloud):
    import onnxruntime as ort
    from lidar_detection.centerpoint import GRID_SIZE
    model=CenterPoint(folder)
    points=np.fromfile(cloud,np.float32).reshape(-1,4)[:,:3].copy() if cloud else np.array([[1,1,0],[1.1,1.1,1],[4,2,.5]],np.float32)
    features,indices,_=pillarize(points)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    providers=[('CUDAExecutionProvider',{'cudnn_conv_algo_search':'HEURISTIC'}),'CPUExecutionProvider'] if model.provider=='CUDAExecutionProvider' else ['CPUExecutionProvider']
    encoder=ort.InferenceSession(str(Path(folder)/'pts_voxel_encoder.onnx'),sess_options=options,providers=providers)
    head=ort.InferenceSession(str(Path(folder)/'pts_backbone_neck_head.onnx'),sess_options=options,providers=providers)
    vectors=encoder.run(None,{encoder.get_inputs()[0].name:features})[0][:,0,:]
    # Independent direct NCHW scatter, matching Autoware scatter_kernel.cu.
    spatial=np.zeros((1,32,GRID_SIZE,GRID_SIZE),np.float32)
    for index,vector in zip(indices,vectors):spatial[0,:,index[1],index[2]]=vector
    reference=head.run(None,{head.get_inputs()[0].name:spatial})
    fused=model.session.run(None,{'vfe/input_features':features,'voxel_indices':indices})
    for a,b in zip(reference,fused):np.testing.assert_allclose(a,b,rtol=2e-4,atol=1e-4)
    for _ in range(3):boxes,stats=model.infer(points)
    print('Original encoder + direct scatter + head == portable fused model; stats:',stats,'boxes:',len(boxes))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir');parser.add_argument('--cloud');args=parser.parse_args()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrePostTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.model_dir:model_check(args.model_dir,args.cloud)

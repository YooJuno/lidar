#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Independent fixtures for original CenterPoint coordinates, features and NMS.
Optional --model-dir checks fused-vs-original neural graph numerically on a
real cloud (--cloud KITTI_XYZI.bin), then reports actual inference timings.
"""
import argparse
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
from lidar_detection.centerpoint import pillarize,decode,rotated_iou,suppress,CenterPoint
from lidar_detection.worker import read_into,reply,RESPONSE
from lidar_detection.native import backend_name
from reference_centerpoint import pillarize as reference_pillarize,rotated_iou as reference_iou

class PrePostTests(unittest.TestCase):
    def test_native_and_numpy_match_frozen_features_bit_for_bit(self):
        rng=np.random.default_rng(71)
        dense=rng.normal([1,2,1],[.03,.03,.1],(6000,3)).astype(np.float32)
        random=rng.uniform([-76.8,-76.8,-4],[76.8,76.8,6],(128000,3)).astype(np.float32)
        x,y=np.meshgrid(np.arange(220),np.arange(200))
        sparse=np.column_stack([(x.ravel()+10.5)*.32-76.8,(y.ravel()+10.5)*.32-76.8,np.zeros(x.size)]).astype(np.float32)
        edges=np.array([[-76.8,-76.8,-4],[76.8,0,0],[0,76.8,0],[0,0,6],
            [0,np.nan,0],[np.inf,0,0],[-np.inf,0,0],[0,0,0],[-0.,0.,0.]],np.float32)
        # Include noncontiguous / float64 inputs as accepted by the public API.
        for points in [dense,random,sparse,edges,random[::3],dense.astype(np.float64),np.empty((0,3),np.float32)]:
            expected=reference_pillarize(points)
            actual=pillarize(points)
            with patch('lidar_detection.native.library',return_value=None):fallback=pillarize(points)
            for result in [actual,fallback]:
                np.testing.assert_array_equal(result[0],expected[0]);np.testing.assert_array_equal(result[1],expected[1])
                for key,value in expected[2].items():self.assertEqual(result[2][key],value)
        self.assertGreater(reference_pillarize(sparse)[2]['truncated_pillars'],0)
    def test_short_range_original_ten_features_and_cap(self):
        rng=np.random.default_rng(29)
        x,y=np.meshgrid(np.arange(300),np.arange(240))
        cap=np.column_stack([(x.ravel()+10.5)*.16-51.2,(y.ravel()+10.5)*.16-51.2,np.zeros(x.size)]).astype(np.float32)
        cases=[rng.uniform([-51.2,-51.2,-3],[51.2,51.2,5],(128000,3)).astype(np.float32),
            rng.normal([1,2,3],[.01,.01,.01],(300,3)).astype(np.float32),cap,
            np.array([[-51.2,-51.2,-3],[51.2,0,0],[0,51.2,0],[0,0,5],[0,np.nan,0],[0,0,0]],np.float32),
            np.empty((0,3),np.float32)]
        for points in cases:
            expected=reference_pillarize(points,'short_range');actual=pillarize(points,'short_range')
            with patch('lidar_detection.native.library',return_value=None):fallback=pillarize(points,'short_range')
            for result in [actual,fallback]:
                np.testing.assert_array_equal(result[0],expected[0]);np.testing.assert_array_equal(result[1],expected[1])
                for key,value in expected[2].items():self.assertEqual(result[2][key],value)
        self.assertEqual(reference_pillarize(cap,'short_range')[2]['truncated_pillars'],8000)
    def test_short_range_decoder_geometry(self):
        arrays={name:np.zeros((1,ch,160,160),np.float32) for name,ch in [('heatmap',5),('reg',2),('height',1),('dim',3),('rot',2)]}
        arrays['heatmap'][:]=-20;arrays['heatmap'][0,4,100,80]=2
        arrays['reg'][0,:,100,80]=[.25,.5];arrays['height'][0,0,100,80]=.9
        arrays['dim'][0,:,100,80]=np.log([.6,1.2,1.8]);arrays['rot'][0,:,100,80]=[0,1]
        box=decode(arrays,.5,'short_range')[0]
        np.testing.assert_allclose(box[:7],[.16,13.12,.9,1.2,.6,1.8,-np.pi/2],atol=1e-5)
        with self.assertRaises(ValueError):decode(arrays,.5,'tiny')
    def test_scalar_clipping_matches_vector_reference(self):
        rng=np.random.default_rng(113)
        for _ in range(400):
            a=np.r_[rng.uniform(-70,70,2),1,rng.uniform(.2,15,3),rng.uniform(-np.pi,np.pi),.8,0].astype(np.float32)
            b=a.copy();b[:2]+=rng.uniform(-10,10,2);b[6]=rng.uniform(-np.pi,np.pi)
            self.assertAlmostEqual(rotated_iou(a,b),reference_iou(a,b),places=11)
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
    def test_spatial_nms_matches_exhaustive_search(self):
        rng=np.random.default_rng(17)
        boxes=rng.uniform(-80,80,(350,9)).astype(np.float32)
        boxes[:,2]=1;boxes[:,3:6]=rng.uniform(.4,8,(350,3));boxes[:,6]=rng.uniform(-np.pi,np.pi,350)
        boxes[:,7]=rng.uniform(.1,1,350);boxes[:,8]=rng.integers(0,5,350)
        # Include neighbors straddling negative and positive bucket boundaries.
        boxes[:6,:2]=[[-10.01,-.01],[-9.99,.01],[-.01,9.99],[.01,10.01],[9.99,-10.01],[10.01,-9.99]]
        kept=[]
        for box in boxes[np.argsort(-boxes[:,7],kind='stable')]:
            rejected=False
            for previous in kept:
                distance=float(np.square(box[:2]-previous[:2]).sum())
                if distance<.25 or (distance<100 and reference_iou(box,previous)>.1):
                    rejected=True;break
            if not rejected:kept.append(box)
        np.testing.assert_array_equal(suppress(boxes),np.asarray(kept))
    def test_worker_reuses_receive_storage_and_handles_short_io(self):
        class ShortReader(io.BytesIO):
            def readinto(self,target):return super().readinto(target[:3])
        storage=bytearray(64)
        read_into(ShortReader(b'abcdefgh'),memoryview(storage)[:8])
        read_into(ShortReader(b'xyz'),memoryview(storage)[:3])
        self.assertEqual(storage[:8],b'xyzdefgh')
        with self.assertRaises(EOFError):read_into(ShortReader(b'x'),memoryview(storage)[:3])
        class ShortWriter(io.BytesIO):
            def write(self,data):return super().write(data[:2])
        sink=ShortWriter();boxes=np.arange(18,dtype=np.float32).reshape(2,9)
        reply(sink,b'LDR1',23,boxes,1.5,'ready')
        data=sink.getvalue();self.assertEqual(RESPONSE.unpack_from(data),(b'LDR1',23,2,1.5,5))
        self.assertEqual(data[RESPONSE.size:],b'ready'+boxes.tobytes())

def model_check(folder,cloud,variant,provider):
    import onnxruntime as ort
    model=CenterPoint(folder,provider,variant,use_tf32=False)
    points=np.fromfile(cloud,np.float32).reshape(-1,4)[:,:3].copy() if cloud else np.array([[1,1,0],[1.1,1.1,1],[4,2,.5]],np.float32)
    features,indices,_=pillarize(points,variant)
    options=ort.SessionOptions();options.intra_op_num_threads=2
    providers=[('CUDAExecutionProvider',{'cudnn_conv_algo_search':'HEURISTIC','use_tf32':0}),'CPUExecutionProvider'] if model.provider=='CUDAExecutionProvider' else ['CPUExecutionProvider']
    encoder=ort.InferenceSession(str(Path(folder)/'pts_voxel_encoder.onnx'),sess_options=options,providers=providers)
    head=ort.InferenceSession(str(Path(folder)/'pts_backbone_neck_head.onnx'),sess_options=options,providers=providers)
    vectors=encoder.run(None,{encoder.get_inputs()[0].name:features})[0][:,0,:]
    # Independent direct NCHW scatter, matching Autoware scatter_kernel.cu.
    spatial=np.zeros((1,32,model.spec.grid,model.spec.grid),np.float32)
    for index,vector in zip(indices,vectors):spatial[0,:,index[1],index[2]]=vector
    reference=head.run(None,{head.get_inputs()[0].name:spatial})
    fused=model.session.run(None,{'vfe/input_features':features,'voxel_indices':indices})
    for a,b in zip(reference,fused):np.testing.assert_allclose(a,b,rtol=2e-4,atol=1e-4)
    for _ in range(3):boxes,stats=model.infer(points)
    print('Original encoder + direct scatter + head == portable fused model; stats:',stats,'boxes:',len(boxes))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model-dir');parser.add_argument('--cloud')
    parser.add_argument('--model',choices=['tiny','short_range'],default='tiny')
    parser.add_argument('--provider',choices=['auto','cpu'],default='auto')
    parser.add_argument('--require-native',action='store_true');args=parser.parse_args()
    if args.require_native and backend_name()!='native':raise SystemExit('Native preprocessing library was not found')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PrePostTests))
    if not result.wasSuccessful():raise SystemExit(1)
    if args.model_dir:model_check(args.model_dir,args.cloud,args.model,args.provider)

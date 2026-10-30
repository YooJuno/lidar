# SPDX-License-Identifier: Apache-2.0
"""Pinned original model bundles and their training-time input geometry."""
from dataclasses import dataclass

AUTOWARE_REVISION='494c8171def40bd36cc2feb323e0a5acbfab132b'
AWML_REVISION='9e44a974aae03a196003950898dc909cd0c680ad'
TINY_URL=f'https://huggingface.co/AutowareFoundation/lidar_centerpoint/resolve/{AUTOWARE_REVISION}/'
SHORT_URL='https://download.autoware-ml-model-zoo.tier4.jp/autoware-ml/models/centerpoint/centerpoint-shortrange/t4base/v2.0.0/'


@dataclass(frozen=True)
class Model:
    name: str
    title: str
    folder: str
    minimum: tuple
    maximum: tuple
    voxel: tuple
    grid: int
    max_voxels: int
    features: int
    downsample: int
    default_score: float  # conservative application preset; not vendor calibration
    files: tuple  # local name, URL, SHA-256

    @property
    def fused_name(self):return self.folder+'-portable-v2.onnx'


MODELS={
    'tiny':Model('tiny','CenterPoint tiny / Autoware v4.1','centerpoint-tiny-v4.1',
        (-76.8,-76.8,-4.),(76.8,76.8,6.),(.32,.32,10.),480,40000,9,2,.35,(
            ('pts_voxel_encoder.onnx',TINY_URL+'tiny/pts_voxel_encoder.onnx','2c53465715c1fd2e9dc5727ef3fca74f4cdf0538f74286b0946e219d0ca5693b'),
            ('pts_backbone_neck_head.onnx',TINY_URL+'tiny/pts_backbone_neck_head.onnx','9bb0b634f3664bd098ce7d6a3d8a9fb7cc8d9b8252b27f302c71e43316bab551'),
            ('MODEL_CARD.md',TINY_URL+'README.md','3cbb8f0f196ef999d3406d88b0d349b808e3d826e2a02a3860dcce94762fc4cf'))),
    'short_range':Model('short_range','AWML ShortRange / ConvNeXt 2.0','centerpoint-shortrange-v2.0.0',
        (-51.2,-51.2,-3.),(51.2,51.2,5.),(.16,.16,8.),640,64000,10,4,.50,(
            ('pts_voxel_encoder.onnx',SHORT_URL+'pts_voxel_encoder_centerpoint_short_range.onnx','c5cfa3e282722f0dd82c8d58294fa3001b0b6d61f38bd48c1e52ee76059ffb85'),
            ('pts_backbone_neck_head.onnx',SHORT_URL+'pts_backbone_neck_head_centerpoint_short_range.onnx','6ac920e99a86a3fabc9e92f0d71e81adbdc2906cedd8e6a17ed0d4207159fe93'),
            ('MODEL_CARD.md',f'https://raw.githubusercontent.com/tier4/AWML/{AWML_REVISION}/projects/CenterPoint/docs/CenterPoint-ShortRange/v2/base.md','027d2f29ced8e1bf30c0a017bac020d7b78e6828139d488ffd36ee065ee9f786'),
            ('LICENSE',f'https://raw.githubusercontent.com/tier4/AWML/{AWML_REVISION}/LICENSE','c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'))),
}


def get_model(name):
    try:return MODELS[name]
    except KeyError:raise ValueError('Unknown detection model: '+str(name)) from None

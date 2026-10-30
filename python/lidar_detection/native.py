# SPDX-License-Identifier: Apache-2.0
"""Optional portable preprocessing; NumPy remains the complete fallback."""
import ctypes
from functools import lru_cache
from pathlib import Path
import sys
import numpy as np


@lru_cache(maxsize=1)
def library():
    name=('lidar_viewer_native.dll' if sys.platform=='win32' else
        'liblidar_viewer_native.dylib' if sys.platform=='darwin' else 'liblidar_viewer_native.so')
    package=Path(__file__).resolve().parent
    # Installed packages keep the helper next to this module. A source build
    # can use any build directory beside the checkout (including multi-config).
    candidates=[package/name]
    # The viewer passes its own location to support arbitrary build paths.
    import os
    native_directory=os.environ.get('LIDAR_VIEWER_NATIVE_DIR')
    if native_directory:candidates.append(Path(native_directory)/name)
    source=package.parent.parent
    candidates.extend([source/'build'/name,source/'build'/'Release'/name,
        source/'build'/'RelWithDebInfo'/name])
    for path in candidates:
        if not path.is_file():continue
        try:
            loaded=ctypes.CDLL(str(path));loaded.lidar_pillar_abi.restype=ctypes.c_int
            if loaded.lidar_pillar_abi()!=2:continue
            loaded.lidar_pillar_order_v2.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_uint32]
            loaded.lidar_pillar_order_v2.restype=ctypes.c_int
            loaded.lidar_pillar_features_v2.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,
                ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,
                ctypes.c_uint32,*([ctypes.c_float]*5),ctypes.c_uint32]
            loaded.lidar_pillar_features_v2.restype=ctypes.c_int
            return loaded
        except (OSError,AttributeError):continue
    return None


def backend_name(point_count=0):return 'native' if point_count<=4_000_000 and library() else 'numpy'


def stable_order(keys,permutation,grid=480):
    keys=np.ascontiguousarray(keys,dtype=np.int32)
    permutation=np.ascontiguousarray(permutation,dtype=np.int64)
    if keys.ndim!=1 or permutation.shape!=keys.shape:raise ValueError('Mismatched pillar ordering arrays')
    native=library()
    if native is None or len(keys)>4_000_000:
        return permutation[np.argsort(keys[permutation],kind='stable')]
    output=np.empty(len(keys),np.int64)
    if native.lidar_pillar_order_v2(keys.ctypes.data,permutation.ctypes.data,len(keys),output.ctypes.data,grid):
        raise ValueError('Invalid native pillar ordering input')
    return output


def fill_features(points,order,starts,counts,keys,features,grid=480,voxel=.32,minimum=(-76.8,-76.8,-4.),voxel_z=10.):
    native=library()
    if native is None or len(points)>4_000_000:return False
    starts=np.ascontiguousarray(starts,dtype=np.int64)
    counts=np.ascontiguousarray(counts,dtype=np.int64)
    keys=np.ascontiguousarray(keys,dtype=np.int32)
    if (points.dtype!=np.float32 or points.ndim!=2 or points.shape[1]!=3 or not points.flags.c_contiguous or
        order.dtype!=np.int64 or order.shape!=(len(points),) or not order.flags.c_contiguous or
        starts.ndim!=1 or starts.shape!=counts.shape or starts.shape!=keys.shape or
        features.dtype!=np.float32 or features.shape[:2]!=(len(keys),32) or features.ndim!=3 or features.shape[2] not in [9,10] or
        not features.flags.c_contiguous or not features.flags.writeable):
        raise ValueError('Invalid native pillar feature arrays')
    if native.lidar_pillar_features_v2(points.ctypes.data,order.ctypes.data,len(points),
            starts.ctypes.data,counts.ctypes.data,keys.ctypes.data,len(keys),features.ctypes.data,
            grid,voxel,*minimum,voxel_z,features.shape[2]):
        raise ValueError('Invalid native pillar feature input')
    return True

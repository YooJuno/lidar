#!/usr/bin/env python3
"""Exercise the packaged C ABI on all OSes using only Python's standard library."""
from array import array
from concurrent.futures import ThreadPoolExecutor
import ctypes
import random
import struct
import sys


def f32(value):return struct.unpack('f',struct.pack('f',value))[0]


def main():
    lib=ctypes.CDLL(sys.argv[1]);assert lib.lidar_pillar_abi()==1
    lib.lidar_pillar_order.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    lib.lidar_pillar_features.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,
        ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
    def address(value):return value.buffer_info()[0]
    assert array('i').itemsize==4 and array('q').itemsize==8 and array('f').itemsize==4
    assert lib.lidar_pillar_order(None,None,0,None)==0
    assert lib.lidar_pillar_order(None,None,4_000_001,None)!=0
    assert lib.lidar_pillar_features(None,None,0,None,None,None,0,None)==0
    def sort_check(seed):
        rng=random.Random(seed);keys=array('i',(rng.randrange(230400) for _ in range(8192)))
        permutation=list(range(len(keys)));rng.shuffle(permutation);permutation=array('q',permutation)
        output=array('q',[-777]*(len(keys)+2))
        assert lib.lidar_pillar_order(address(keys),address(permutation),len(keys),address(output)+8)==0
        assert output[0]==output[-1]==-777
        assert output[1:-1].tolist()==sorted(permutation,key=lambda i:keys[i])
        return True
    with ThreadPoolExecutor(max_workers=4) as pool:assert all(pool.map(sort_check,range(16)))
    key=246*480+243
    for size in [1,2,40]:
        points=array('f',(v for rank in range(size) for v in [1+rank*.001,2+rank*.001,3+rank*.01]))
        order=array('q',reversed(range(size)));starts=array('q',[0]);counts=array('q',[size]);keys=array('i',[key])
        features=array('f',[0]*(32*9+2));features[0]=features[-1]=12345
        assert lib.lidar_pillar_features(address(points),address(order),size,address(starts),address(counts),address(keys),1,address(features)+4)==0
        assert features[0]==features[-1]==12345
        used=min(size,32);mean=[f32(f32(sum(points[order[r]*3+a] for r in range(used)))/used) for a in range(3)]
        center=[f32(f32(f32(v+.5)*f32(.32))-f32(76.8)) for v in [243,246]]
        expected=[]
        for rank in range(32):
            if rank>=used:expected.extend([0]*9);continue
            p=points[order[rank]*3:order[rank]*3+3]
            expected.extend([*p,0,*[f32(p[a]-mean[a]) for a in range(3)],f32(p[0]-center[0]),f32(p[1]-center[1])])
        assert features[1:-1].tobytes()==array('f',expected).tobytes()
        starts[0]=-1
        assert lib.lidar_pillar_features(address(points),address(order),size,address(starts),address(counts),address(keys),1,address(features)+4)!=0
    keys=array('i',[-1]);permutation=array('q',[0]);output=array('q',[999])
    assert lib.lidar_pillar_order(address(keys),address(permutation),1,address(output))!=0 and output[0]==999
    print('detector_native_smoke: stable sorting, float32 features, padding, truncation, bounds and concurrent reuse passed')
    return 0


if __name__=='__main__':raise SystemExit(main())

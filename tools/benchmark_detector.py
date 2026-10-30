#!/usr/bin/env python3
"""Measure the installed detector on one KITTI XYZI scan without a GUI."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time


def resident_mib():
    try:
        rows=Path('/proc/self/status').read_text().splitlines()
        return int(next(row.split()[1] for row in rows if row.startswith('VmRSS:')))/1024
    except (OSError,StopIteration):return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--cloud',type=Path,required=True)
    parser.add_argument('--provider',choices=['auto','cpu'],default='auto')
    parser.add_argument('--iterations',type=int,default=50)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if not 1<=args.iterations<=10000:parser.error('--iterations must be 1..10000')
    import numpy as np
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'python'))
    from lidar_detection.centerpoint import CenterPoint
    points=np.fromfile(args.cloud,np.float32).reshape(-1,4)[:,:3].copy()
    started=time.perf_counter();model=CenterPoint(args.model_dir,args.provider)
    startup=time.perf_counter()-started
    for _ in range(5):model.infer(points,.1)
    rss_warm=resident_mib();timings=[];rss_samples=[]
    for index in range(args.iterations):
        boxes,stats=model.infer(points,.1);timings.append(stats)
        if (index+1)%10==0:rss_samples.append(resident_mib())
    keys=['preprocess_ms','inference_ms','postprocess_ms','latency_ms']
    result={'platform':platform.platform(),'provider':model.provider,'preprocess_backend':stats['preprocess_backend'],'points':len(points),
        'iterations':args.iterations,'startup_s':startup,'cached_boxes':len(boxes),
        'p50_ms':{key:float(np.median([value.get(key,0) for value in timings])) for key in keys},
        'p95_pipeline_ms':float(np.percentile([value['latency_ms'] for value in timings],95)),
        'rss_after_warmup_mib':rss_warm,'rss_final_mib':resident_mib(),'rss_samples_mib':rss_samples,
        'torch_imported':'torch' in sys.modules}
    text=json.dumps(result,indent=2);print(text)
    if args.output:args.output.write_text(text+'\n')
    return 0


if __name__=='__main__':raise SystemExit(main())

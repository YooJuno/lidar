#!/usr/bin/env python3
"""Validate/run an installed Studio adapter; never invoke a shell."""
import argparse
import importlib
import json
import math
from pathlib import Path
import sys
import traceback


def load_request(path):
    data=json.loads(path.read_text())
    for key in ('epochs','max_points','seed','batch_size','workers'):data[key]=int(data.get(key,{'epochs':30,'max_points':4096,'seed':42,'batch_size':2,'workers':2}[key]))
    for key,default in [('learning_rate',.001),('validation_fraction',.2),('yaw_range',25.)]:data[key]=float(data.get(key,default))
    data['allow_weak']=data.get('allow_weak',False)in (True,'true','1')
    if not 1<=data['epochs']<=10000 or not 32<=data['max_points']<=200000 or not 1<=data['batch_size']<=256 or not 0<=data['workers']<=64:raise ValueError('Training parameter is out of range')
    if not 0<=data['seed']<2**32:raise ValueError('Random seed must be a nonnegative 32-bit integer')
    if not all(math.isfinite(data[k])for k in ('learning_rate','validation_fraction','yaw_range'))or not 0<data['learning_rate']<1 or not 0<=data['yaw_range']<=180:raise ValueError('Invalid learning rate or augmentation')
    data['project']=str(Path(data['project']).expanduser().resolve());return data


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True);parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    output=args.request.resolve().parent
    try:
        request=load_request(args.request)
        sys.path.insert(1,request['project'])
        profiles=json.loads((Path(__file__).parent/'lidar_training/adapters.json').read_text())
        profile=next((p for p in profiles if p['id']==request['adapter']),None)
        if not profile:raise ValueError('Unknown training adapter')
        adapter=importlib.import_module(profile['module']);ready=adapter.validate(request)
        print('Validation passed:',json.dumps(ready,ensure_ascii=False),flush=True)
        (output/'validation.json').write_text(json.dumps(ready,ensure_ascii=False,indent=2)+'\n')
        if not args.validate:adapter.run(request,output)
        (output/'result.json').write_text(json.dumps(dict(status='validated'if args.validate else'complete',adapter=request['adapter']))+'\n')
        return 0
    except Exception as error:
        print('Training setup failed:',error,flush=True);traceback.print_exc()
        (output/'result.json').write_text(json.dumps(dict(status='failed',error=str(error)))+'\n');return 1


if __name__=='__main__':raise SystemExit(main())

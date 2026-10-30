#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Install the optional original Autoware CenterPoint model in user storage."""
import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys
import urllib.request
import venv

REVISION='494c8171def40bd36cc2feb323e0a5acbfab132b'
FILES={
 'MODEL_CARD.md':'3cbb8f0f196ef999d3406d88b0d349b808e3d826e2a02a3860dcce94762fc4cf',
 'pts_voxel_encoder.onnx':'2c53465715c1fd2e9dc5727ef3fca74f4cdf0538f74286b0946e219d0ca5693b',
 'pts_backbone_neck_head.onnx':'9bb0b634f3664bd098ce7d6a3d8a9fb7cc8d9b8252b27f302c71e43316bab551',
}

def directories():
    home=Path.home()
    config=home/'.config/lidar_viewer'
    data_base=Path(os.environ.get('XDG_DATA_HOME',home/'.local/share'))
    data=(data_base if data_base.is_absolute() else home/'.local/share')/'lidar_viewer'
    if sys.platform=='win32':
        config=Path(os.environ.get('APPDATA',home/'AppData/Roaming'))/'LiDARViewer'
        data=Path(os.environ.get('LOCALAPPDATA',home/'AppData/Local'))/'LiDARViewer'
    elif sys.platform=='darwin':config=data=home/'Library/Application Support/LiDARViewer'
    # Match the viewer's explicit XDG_CONFIG_HOME override on every platform.
    if os.environ.get('XDG_CONFIG_HOME') and Path(os.environ['XDG_CONFIG_HOME']).is_absolute():
        config=Path(os.environ['XDG_CONFIG_HOME'])/'lidar_viewer'
    return config,data/'detector'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda',action='store_true',help='ONNX Runtime CUDA 12 + cuDNN 9 (Linux/Windows x64)')
    parser.add_argument('--runtime-python',type=Path,help='Use an existing isolated Python with numpy, onnx and onnxruntime; install no packages')
    args=parser.parse_args()
    if not args.runtime_python and not (3,10)<=sys.version_info[:2]<=(3,13):
        parser.error('Use Python 3.10 through 3.13 for the pinned model runtime')
    if args.cuda and (sys.platform=='darwin' or platform.machine().lower() not in ('x86_64','amd64')):
        parser.error('CUDA runtime requires Linux or Windows x64')
    config,data=directories();model=data/'centerpoint-tiny-v4.1';model.mkdir(parents=True,exist_ok=True)
    if args.runtime_python:
        python=args.runtime_python.expanduser().absolute()
    else:
        runtime=data/'runtime'
        print('Creating isolated Python runtime:',runtime,flush=True)
        venv.EnvBuilder(with_pip=True,system_site_packages=False).create(runtime)
        python=runtime/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
        runtime_package='onnxruntime-gpu[cuda,cudnn]==1.23.2' if args.cuda else 'onnxruntime==1.23.2'
        subprocess.run([str(python),'-m','pip','install','--disable-pip-version-check','numpy==2.2.6','onnx==1.20.1',runtime_package],check=True)
    subprocess.run([str(python),'-c','import numpy,onnx,onnxruntime'],check=True)
    for name,digest in FILES.items():
        target=model/name
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()==digest:continue
        print('Downloading original Autoware weight:',name,flush=True)
        temporary=target.with_suffix('.download')
        try:
            remote='README.md' if name=='MODEL_CARD.md' else 'tiny/'+name
            request=urllib.request.Request(f'https://huggingface.co/AutowareFoundation/lidar_centerpoint/resolve/{REVISION}/{remote}',headers={'User-Agent':'LiDAR-Studio-model-setup'})
            with urllib.request.urlopen(request,timeout=120) as source,temporary.open('wb') as output:
                while True:
                    chunk=source.read(1024*1024)
                    if not chunk:break
                    output.write(chunk)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:raise RuntimeError('Model checksum mismatch: '+name)
            temporary.replace(target)
        finally:temporary.unlink(missing_ok=True)
    root=Path(__file__).resolve().parent.parent
    python_root=root/'python'
    if not (python_root/'lidar_detection').is_dir():raise RuntimeError('Cannot locate the packaged detector adapter')
    # Validate the actual model and provider before publishing configuration.
    subprocess.run([str(python),'-c',
        'import sys;sys.path.insert(0,sys.argv[1]);from lidar_detection.centerpoint import CenterPoint;'
        'from lidar_detection.centerpoint import fuse_model;fuse_model(sys.argv[2]);'
        'm=CenterPoint(sys.argv[2],sys.argv[3]);print("Validated:",m.provider)',
        str(python_root),str(model),'auto' if args.cuda else 'cpu'],check=True)
    config.mkdir(parents=True,exist_ok=True)
    text=f'python={python}\nmodel_dir={model}\nprovider={"auto" if args.cuda else "cpu"}\n'
    if '\n' in str(python) or '\n' in str(model):raise RuntimeError('Newlines in runtime paths are unsupported')
    temporary=config/'detector.cfg.tmp';temporary.write_text(text,encoding='utf-8');temporary.replace(config/'detector.cfg')
    print('Ready. Open Objects > Enable objects (or Retry model). Configuration:',config/'detector.cfg')

if __name__=='__main__':main()

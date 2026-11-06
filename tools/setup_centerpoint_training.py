#!/usr/bin/env python3
"""Install/register an isolated MMDetection3D CenterPoint CUDA trainer (Linux x64)."""
import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

REVISION='fe25f7a51d36e3702f961e198894580d83c4387b'  # MMDetection3D v1.4.0
CONFIG='configs/centerpoint/centerpoint_pillar02_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py'
MMCV='https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/mmcv-2.1.0-cp311-cp311-manylinux1_x86_64.whl'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path(os.environ.get('XDG_DATA_HOME',Path.home()/'.local/share'))
    parser.add_argument('--root',type=Path,default=base/'lidar_viewer/training')
    parser.add_argument('--python',default='python3.11',help='Python 3.11 used to create the training environment')
    parser.add_argument('--register-only',action='store_true',help='Check an existing installation and register it; install nothing')
    args=parser.parse_args()
    if sys.platform!='linux' or platform.machine().lower()not in('x86_64','amd64'):parser.error('This pinned trainer installer supports Linux x64 with an NVIDIA CUDA GPU')
    root=args.root.expanduser().resolve();runtime=root/'centerpoint-env';project=root/'mmdetection3d';python=runtime/'bin/python'
    def run(*command):subprocess.run([str(item)for item in command],check=True)
    if not args.register_only:
        version=subprocess.check_output([args.python,'-c','import sys; print("%d.%d"%sys.version_info[:2])'],text=True).strip()
        if version!='3.11':parser.error('--python must be Python 3.11 for the pinned CUDA wheels')
        root.mkdir(parents=True,exist_ok=True)
        if not python.exists():run(args.python,'-m','venv',runtime)
        def pip(*packages):run(python,'-m','pip','install','--disable-pip-version-check',*packages)
        pip('torch==2.1.2','torchvision==0.16.2','--index-url','https://download.pytorch.org/whl/cu118')
        pip('numpy==1.26.4','setuptools<81','wheel','yapf==0.40.1','opencv-python==4.10.0.84','mmengine==0.10.7','mmdet==3.3.0',MMCV)
        if not project.exists():run('git','clone','--depth','1','--branch','v1.4.0','https://github.com/open-mmlab/mmdetection3d.git',project)
        revision=subprocess.check_output(['git','-C',str(project),'rev-parse','HEAD'],text=True).strip()
        if revision!=REVISION:raise RuntimeError('Existing trainer source differs from the pinned revision; choose another --root')
        pip('--no-build-isolation','-e',str(project),'numpy==1.26.4','opencv-python==4.10.0.84')
    if not(project/CONFIG).is_file():raise RuntimeError('CenterPoint base config is missing')
    run(python,'-c','import torch,mmcv,mmdet,mmdet3d; from mmdet3d.utils import register_all_modules; register_all_modules(init_default_scope=True); assert torch.cuda.is_available(), "CUDA GPU unavailable"; print("Trainer ready:",torch.__version__,mmcv.__version__,mmdet.__version__,mmdet3d.__version__)')
    config=Path(os.environ.get('XDG_CONFIG_HOME',Path.home()/'.config'))/'lidar_viewer';config.mkdir(parents=True,exist_ok=True)
    target=config/'training.json';settings=json.loads(target.read_text())if target.is_file()else{}
    profile=settings.setdefault('profiles',{}).setdefault('centerpoint',{})
    profile.update(python=str(python),project=str(project),config=str(project/CONFIG))
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(settings,indent=2)+'\n');temporary.replace(target)
    print('Registered. Open Train > CenterPoint. Runtime:',python)


if __name__=='__main__':main()

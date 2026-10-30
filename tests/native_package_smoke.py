#!/usr/bin/env python3
"""Verify the shipped helper, including loading its C ABI after extraction."""
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packages',type=Path,required=True);args=parser.parse_args()
    native=('lidar_viewer_native.dll' if sys.platform=='win32' else
        'liblidar_viewer_native.dylib' if sys.platform=='darwin' else 'liblidar_viewer_native.so')
    archives=sorted([*args.packages.glob('*.zip'),*args.packages.glob('*.tar.gz')])
    if not archives:raise RuntimeError('No portable package archive found')
    for archive in archives:
        with tempfile.TemporaryDirectory(prefix='lidar-native-package-') as temporary:
            if archive.name.endswith('.zip'):
                with zipfile.ZipFile(archive) as source:
                    matches=[name for name in source.namelist() if name.endswith('/python/lidar_detection/'+native)]
                    if len(matches)!=1:raise RuntimeError(f'{archive.name}: missing or ambiguous native helper')
                    data=source.read(matches[0])
            else:
                with tarfile.open(archive,'r:gz') as source:
                    matches=[member for member in source.getmembers() if member.name.endswith('/python/lidar_detection/'+native)]
                    if len(matches)!=1:raise RuntimeError(f'{archive.name}: missing or ambiguous native helper')
                    data=source.extractfile(matches[0]).read()
            library=Path(temporary)/native;library.write_bytes(data)
            subprocess.run([sys.executable,str(Path(__file__).with_name('detector_native_smoke.py')),str(library)],check=True)
            print('Verified packaged native helper:',archive.name)
    return 0


if __name__=='__main__':raise SystemExit(main())

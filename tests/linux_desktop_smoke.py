#!/usr/bin/env python3
"""Exercise desktop registration and launch arguments in private user directories."""
import configparser
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='lidar-desktop-') as temporary:
    folder = Path(temporary) / "space ' $literal % 한글"
    folder.mkdir()
    binary = folder / "viewer ' ; touch UNEXPECTED"
    binary.write_text('#!/usr/bin/python3\nimport json,sys,os\n'
                      'from pathlib import Path\n'
                      'value=json.dumps(sys.argv[1:])\n'
                      'if os.getenv("LIDAR_DESKTOP_SMOKE_RESULT"):Path(os.environ["LIDAR_DESKTOP_SMOKE_RESULT"]).write_text(value)\n'
                      'else:print(value)\n')
    binary.chmod(0o755)
    data = folder / 'share'
    bins = folder / 'bin'
    command = [sys.executable, str(root / 'tools/install_linux_desktop.py'), '--binary', str(binary),
               '--data-home', str(data), '--bin-dir', str(bins)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    entry = data / 'applications/lidar-viewer.desktop'
    settings = configparser.ConfigParser(interpolation=None)
    settings.read(entry)
    assert settings['Desktop Entry']['StartupWMClass'] == 'lidar-viewer'
    assert settings['Desktop Entry']['Name'] == 'LiDAR Studio'
    icon = data / 'icons/hicolor/scalable/apps/lidar-viewer.svg'
    assert icon.read_bytes() == (root / 'resources/icons/lidar-studio.svg').read_bytes()
    arguments = ['--detect', 'bag file $literal.mcap', 'quotes " \' % 字']
    result = subprocess.run([str(bins / 'lidar-viewer'), *arguments], cwd=folder,
                            check=True, text=True, capture_output=True)
    assert json.loads(result.stdout) == arguments
    assert not (folder / 'UNEXPECTED').exists()
    subprocess.run(command, check=True, capture_output=True)  # Safe to update after rebuilding.
    # Actual GLib Desktop Entry parser/launcher when PyGObject is available.
    try:
        from gi.repository import Gio
    except ImportError:
        pass
    else:
        app = Gio.DesktopAppInfo.new_from_filename(str(entry))
        assert app and app.get_startup_wm_class() == 'lidar-viewer'
        assert Path(app.get_icon().get_file().get_path()) == icon
        # Check the real Exec field through GLib, including quotes/$/% in paths.
        marker = folder / 'launched.json'
        context = Gio.AppLaunchContext()
        context.setenv('LIDAR_DESKTOP_SMOKE_RESULT', str(marker))
        app.launch([], context)
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert json.loads(marker.read_text()) == []
    entry.write_text('[Desktop Entry]\nName=Personal launcher\n')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'unrelated file' in result.stderr
    assert 'Personal launcher' in entry.read_text()
print('Linux desktop registration, icon, safe command quoting and update checks passed.')

#!/usr/bin/env python3
"""Register a source build in the Linux application menu and dock, without sudo."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MARKER = '# Managed by LiDAR Studio: install_linux_desktop.py'


def desktop_string(value):
    return str(value).replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')


def desktop_exec(path):
    # Desktop Entry quoting has two layers: string escapes, then Exec escapes.
    value = str(path).replace('%', '%%')
    for character in ('\\', '"', '$', '`'):
        value = value.replace(character, '\\' + character)
    return desktop_string('"' + value + '"')


def write_managed(path, content, mode):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and MARKER not in path.read_text(encoding='utf-8'):
        raise RuntimeError(f'Refusing to overwrite an unrelated file: {path}')
    handle, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(content)
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=ROOT / 'build/lidar_viewer')
    parser.add_argument('--data-home', type=Path, help='Override XDG_DATA_HOME')
    parser.add_argument('--bin-dir', type=Path, help='Override ~/.local/bin')
    args = parser.parse_args()
    if not sys.platform.startswith('linux'):
        parser.error('This launcher is for Linux; Windows and macOS use native package icons.')
    binary = args.binary.expanduser().resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error('Build the viewer first, or pass --binary /absolute/path/to/lidar_viewer')
    if args.data_home is not None:
        data_home = args.data_home.expanduser().absolute()
    else:
        data_home = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')).expanduser()
        if not data_home.is_absolute():
            data_home = Path.home() / '.local/share'
    bin_dir = (args.bin_dir or Path.home() / '.local/bin').expanduser().absolute()
    launcher = bin_dir / 'lidar-viewer'
    entry = data_home / 'applications/lidar-viewer.desktop'
    icon = data_home / 'icons/hicolor/scalable/apps/lidar-viewer.svg'
    # Check ownership of both text files before making any changes.
    for path in (launcher, entry):
        if path.exists() and MARKER not in path.read_text(encoding='utf-8'):
            raise RuntimeError(f'Refusing to overwrite an unrelated file: {path}')
    template = (ROOT / 'resources/lidar-viewer.desktop').read_text(encoding='utf-8')
    lines = [MARKER]
    for line in template.splitlines():
        if line.startswith('Exec='):
            line = 'Exec=/bin/bash ' + desktop_exec(launcher)
        elif line.startswith('Icon='):
            # An absolute icon also works before GNOME refreshes its theme cache.
            line = 'Icon=' + desktop_string(icon)
        lines.append(line)
    script = '#!/usr/bin/env bash\n' + MARKER + '\nset -e\n'
    script += 'if [[ -r /opt/ros/jazzy/setup.bash ]]; then\n  source /opt/ros/jazzy/setup.bash\nfi\n'
    script += 'exec ' + shlex.quote(str(binary)) + ' "$@"\n'
    icon.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / 'resources/icons/lidar-studio.svg', icon)
    write_managed(launcher, script, 0o755)
    write_managed(entry, '\n'.join(lines) + '\n', 0o644)
    updater = shutil.which('update-desktop-database')
    if updater:
        subprocess.run([updater, str(entry.parent)], check=True)
    print('Registered LiDAR Studio for the application menu and dock.')
    print('Desktop entry:', entry)
    print('Launcher:', launcher)


if __name__ == '__main__':
    main()

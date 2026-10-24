"""Discover external runtimes; all project files are relative to this checkout."""
from pathlib import Path
import os, shutil
from PIL import ImageFont

PROJECT_ROOT=Path(__file__).resolve().parents[1]

def octave_runtime():
    configured=os.environ.get('OCTAVE_EXECUTABLE')
    candidates=[configured] if configured else [shutil.which('octave-cli'),shutil.which('octave-cli.exe'),
        'C:/Octave/mingw64/bin/octave-cli.exe']
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            exe=Path(candidate).resolve()
            env=os.environ.copy()
            env['PATH']=os.pathsep.join([str(exe.parent),str(exe.parent.parent/'usr'/'bin'),env.get('PATH','')])
            env['NAOMI_ROOT']=str(PROJECT_ROOT/'vendor'/'NAOMi_octave')
            env['OPENBLAS_NUM_THREADS']='1'
            return str(exe),env
    raise RuntimeError('GNU Octave not found. Install Octave and set OCTAVE_EXECUTABLE to octave-cli.exe (see README_RU.md). Cached baseline seeds do not require Octave.')

def plot_font(size):
    candidates=[os.environ.get('NAOMIGEN_FONT'),'C:/Windows/Fonts/arial.ttf',
       '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf','DejaVuSans.ttf']
    for candidate in candidates:
        if not candidate:continue
        try:return ImageFont.truetype(candidate,size)
        except OSError:pass
    return ImageFont.load_default(size=size)

"""Read back exported files with Pillow/FFmpeg, independently of Octave."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image


def validate(folder: Path) -> dict:
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    cfg = metadata["cfg"]
    expected_frames = int(cfg["n_frames"])
    maxima = None
    image_shape = None
    for filename in ("activity_noisy.tif", "activity_clean.tif"):
        with Image.open(folder / filename) as image:
            assert image.n_frames == expected_frames, (filename, image.n_frames)
            for frame in range(expected_frames):
                image.seek(frame)
                array = np.asarray(image)
                assert array.dtype == np.uint16, (filename, array.dtype)
                image_shape = array.shape
                if filename == "activity_clean.tif":
                    maxima = array.copy() if maxima is None else np.maximum(maxima, array)
    with Image.open(folder / "preview.png") as image:
        assert np.array_equal(np.asarray(image), maxima), "PNG differs from clean TIFF maximum"

    completed = subprocess.run(
        [r"C:\Octave\mingw64\bin\ffprobe.exe", "-v", "error", "-count_frames",
         "-select_streams", "v:0", "-show_entries",
         "stream=codec_name,width,height,r_frame_rate,nb_read_frames:format=duration",
         "-of", "json", str(folder / "activity_noisy.avi")],
        check=True, capture_output=True, text=True, encoding="utf-8",
    )
    probe = json.loads(completed.stdout)
    stream = probe["streams"][0]
    assert int(stream["nb_read_frames"]) == expected_frames
    assert abs(float(Fraction(stream["r_frame_rate"])) - cfg["fps"]) < 1e-10
    assert (stream["height"], stream["width"]) == image_shape
    assert abs(float(probe["format"]["duration"]) - expected_frames / cfg["fps"]) < 0.001
    report = {"passed": True, "reader": "Pillow, NumPy, FFprobe",
              "tiff_frames": expected_frames, "frame_shape": image_shape,
              "avi": probe, "sha256": {}}
    for filename in ("simulation_result.mat", "activity_noisy.tif",
                     "activity_clean.tif", "activity_noisy.avi", "preview.png"):
        report["sha256"][filename] = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
    (folder / "export_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.folder), indent=2))

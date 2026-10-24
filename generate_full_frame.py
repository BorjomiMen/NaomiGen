r"""Reference-conditioned 400x485 still images from individual NAOMi footprints.

Run: .\.venv\Scripts\python.exe .\generate_full_frame.py --count 4 --seed 42
No tiled 64x64 images, image enlargement, MATLAB, or new Octave run is used.
This composes existing NAOMi optical contributions in 2D; it does not simulate
a single whole-fish 3D volume. Large-scale anatomy is conditioned on real data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STAGE = ROOT / "naomi_zenodo_stage1"


def dependencies():
    """Use the repository environment, including when launched with plain python."""
    import importlib.util
    import os
    local = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if os.name == "nt" and not local.is_file():
        subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                        str(ROOT / "Setup.ps1"), "-Python", sys.executable], check=True)
    if local.is_file() and Path(sys.executable).resolve() != local.resolve():
        raise SystemExit(subprocess.call([str(local), str(Path(__file__).resolve()), *sys.argv[1:]]))
    needed = ("numpy", "scipy", "skimage", "tifffile", "PIL")
    if any(importlib.util.find_spec(name) is None for name in needed):
        if os.name != "nt":
            raise SystemExit("Install dependencies: python -m pip install -r naomi_zenodo_stage1/requirements.txt")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(STAGE / "requirements.txt")], check=True)
        raise SystemExit(subprocess.call([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def detect(image):
    """Same proxy as the local pipeline, with arbitrary image dimensions."""
    from scipy.spatial import cKDTree
    from skimage.feature import peak_local_max
    from skimage.measure import regionprops
    from skimage.segmentation import watershed
    smooth = ndi.gaussian_filter(image.astype(float), .65)
    response = smooth - ndi.gaussian_filter(image.astype(float), 4)
    residual = image - smooth
    noise = float(np.median(np.abs(residual - np.median(residual))) / .67448975)
    peaks = peak_local_max(response, min_distance=3, threshold_abs=max(300, .8 * noise), exclude_border=3)
    markers = np.zeros(image.shape, np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(-smooth, markers, mask=response > max(100, .25 * noise))
    h, w = image.shape
    props = [p for p in regionprops(labels) if 5 <= p.area <= 100 and
             p.bbox[0] > 0 and p.bbox[1] > 0 and p.bbox[2] < h and p.bbox[3] < w]
    centers = np.asarray([p.centroid for p in props], float).reshape(-1, 2)
    diameters = [p.equivalent_diameter_area * .873 for p in props]
    nearest = cKDTree(centers).query(centers, k=2)[0][:, 1] * .873 if len(centers) > 1 else []
    return centers, {"count": len(centers),
                     "diameter_median_um": float(np.median(diameters)) if len(diameters) else None,
                     "nearest_neighbor_median_um": float(np.median(nearest)) if len(nearest) else None}


def image_metrics(image):
    z = image.astype(float)
    return {"mean": float(z.mean()), "std": float(z.std()),
            "p10": float(np.percentile(z, 10)), "p50": float(np.percentile(z, 50)),
            "p90": float(np.percentile(z, 90)),
            "saturation_fraction": float((z >= 32767).mean()),
            "gradient_rms": float(np.sqrt(np.mean(np.diff(z, axis=0)**2) + np.mean(np.diff(z, axis=1)**2)))}


def robust_noise(stack):
    differences = np.diff(stack.astype(np.float32), axis=0) / np.sqrt(2)
    return float(np.median(np.abs(differences - np.median(differences))) / .67448975)


def prepare_reference(path, cfg, window):
    from skimage.measure import label, regionprops
    stack = tifffile.imread(path)
    if stack.shape != (128, 400, 485) or not np.isfinite(stack).all():
        raise ValueError("Expected the bundled Zenodo sample: 128 frames, 400 rows x 485 columns.")
    provenance = json.loads((STAGE / "reference/provenance.json").read_text(encoding="utf-8"))
    if sha256(path) != provenance["sample_sha256"]:
        raise ValueError("Reference SHA-256 differs from reference/provenance.json.")
    indices = provenance["source_frame_indices_zero_based"]
    frames = np.asarray(indices)[::16]
    offset = cfg["camera_offset_adu"]
    slope = cfg["log_brightness_slope_per_source_frame"]
    anchor = cfg["brightness_reference_frame"]
    # Only windows 0..4 determine the spatial template. Later windows are not fitted.
    means = np.stack([stack[i*16:(i+1)*16].mean(0) for i in range(5)])
    factors = np.exp(slope * (frames[:5] - anchor))
    fit = offset + np.mean((means - offset) / factors[:, None, None], axis=0)
    centers, morph = detect(fit)
    impulses = np.zeros(fit.shape, np.float32)
    rounded = np.rint(centers).astype(int)
    np.add.at(impulses, (rounded[:, 0], rounded[:, 1]), 1)
    density = ndi.gaussian_filter(impulses, 9)
    fit_signal = np.maximum(fit - offset, 0)
    # Preserve connected dark cavities/vessels and elongated bright non-cell
    # structures. Individual cell texture is removed from the diffuse field.
    bright_labels = label(fit > 28000)
    structural_ids = [p.label for p in regionprops(bright_labels)
                      if p.area >= 80 and p.axis_major_length >= 30 and
                      p.axis_major_length / max(p.axis_minor_length, 1) >= 3]
    bright = ndi.binary_dilation(np.isin(bright_labels, structural_ids), iterations=2)
    valid = (~bright).astype(float)
    low = ndi.gaussian_filter(fit_signal * valid, 8) / np.maximum(ndi.gaussian_filter(valid, 8), .01)
    dark_labels = label(fit_signal < 450)
    dark_ids = [p.label for p in regionprops(dark_labels) if p.area >= 150]
    dark = np.isin(dark_labels, dark_ids)
    structure_weight = np.clip(ndi.gaussian_filter((bright | dark).astype(float), .8), 0, 1)
    structural_field = ndi.gaussian_filter(fit_signal, .8)
    low = np.maximum(low * (1-structure_weight) + structural_field * structure_weight, 0)
    density *= np.clip(low / 800, 0, 1)
    if not density.sum() > 0 or len(centers) < 100:
        raise ValueError("Reference contains too few detectable cells for this preset.")
    # Estimate cell contrast within each 16-frame burst. Averaging distant
    # bursts first would erase cells whose calcium fluorescence changes.
    normalized = offset + (means-offset)/factors[:, None, None]
    variances = [ndi.gaussian_filter((m-ndi.gaussian_filter(m, 4))**2, 9) for m in normalized]
    contrast = np.sqrt(np.maximum(np.mean(variances, axis=0), 1))
    brightness = float(np.exp(slope * (frames[window] - anchor)))
    contrast_factor = float(np.exp(cfg["log_contrast_slope_per_source_frame"] * (frames[window] - anchor)))
    real = stack[16*window:16*(window+1)].copy()
    return {"low": low, "contrast": contrast, "density": density,
            "fit_detected_cells": len(centers), "fit_morphology": morph,
            "brightness": brightness, "contrast_factor": contrast_factor,
            "reference_frame": int(frames[window]), "real": real,
            "reference_sha256": sha256(path), "structural_pixels": int((bright | dark).sum())}


def footprint_bank(cfg, seeds):
    from scipy.io import loadmat
    bank, sources = [], []
    grid = np.arange(-11, 12, dtype=float) * cfg["pixel_size_um"]
    for seed in seeds:
        path = STAGE / f"anatomy/{cfg['anatomy_tag']}_seed{seed}/footprints.mat"
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached NAOMi scene: {path}")
        data = loadmat(path, simplify_cells=True)
        footprints = np.maximum(data["footprints"].astype(np.float32), 0)
        if footprints.ndim != 3 or not np.isfinite(footprints).all():
            raise ValueError(f"Invalid footprints: {path}")
        for cell in range(footprints.shape[-1]):
            raw = footprints[:, :, cell]
            peak = float(raw.max())
            if peak * cfg["gain_adu"] / cfg["footprint_reference_peak"] < 1400:
                continue
            cy, cx = ndi.center_of_mass(np.where(raw >= .5 * peak, raw, 0))
            if not (9 <= cy < raw.shape[0]-9 and 9 <= cx < raw.shape[1]-9):
                continue
            yy, xx = np.meshgrid(cy + grid, cx + grid, indexing="ij")
            patch = ndi.map_coordinates(raw, [yy, xx], order=1, mode="constant", cval=0)
            patch = ndi.gaussian_filter(patch, cfg["additional_blur_px"])
            patch *= cfg["gain_adu"] / cfg["footprint_reference_peak"]
            half_area = int((patch >= .5 * patch.max()).sum())
            if not 5 <= half_area <= 60:
                continue
            patch[patch < patch.max() * .001] = 0
            bank.append(patch.astype(np.float32))
            sources.append({"anatomy_seed": seed, "cell_id": cell + 1})
        print(f"NAOMi seed {seed}: {len(bank)} optical templates accumulated", flush=True)
    if len(bank) < 30:
        raise ValueError("Too few usable NAOMi templates.")
    return np.stack(bank), sources


def positions(density, count, rng):
    """Draw a global inhomogeneous point process, without image tiles."""
    h, w = density.shape
    probabilities = density.ravel().astype(float)
    probabilities /= probabilities.sum()
    selected = rng.choice(h*w, size=count*35, p=probabilities)
    buckets, result = {}, []
    distance = 2.0
    for flat in selected:
        y, x = divmod(int(flat), w)
        y += rng.uniform(-.49, .49)
        x += rng.uniform(-.49, .49)
        by, bx = int(y // distance), int(x // distance)
        nearby = [p for iy in range(by-1, by+2) for ix in range(bx-1, bx+2)
                  for p in buckets.get((iy, ix), ())]
        if any((y-py)**2 + (x-px)**2 < distance**2 for py, px in nearby):
            continue
        result.append((y, x))
        buckets.setdefault((by, bx), []).append((y, x))
        if len(result) == count:
            break
    if len(result) != count:
        raise RuntimeError(f"Could place only {len(result)} of {count} cells; lower --density-factor.")
    return result


def compose(bank, sources, cfg, ref, seed, density_factor):
    rng = np.random.default_rng(seed)
    shape = ref["low"].shape
    n_cells = int(round(ref["fit_detected_cells"] * density_factor))
    if n_cells >= 65535:
        raise ValueError("Too many cells for uint16 instance IDs.")
    centers = positions(ref["density"], n_cells, rng)
    cells = []
    signal = np.zeros(shape, np.float32)
    for y, x in centers:
        template_id = int(rng.integers(len(bank)))
        patch = np.rot90(bank[template_id], int(rng.integers(4)))
        if rng.random() < .5:
            patch = patch[:, ::-1]
        yi, xi = int(round(y)), int(round(x))
        patch = ndi.shift(patch, (y-yi, x-xi), order=1, mode="constant", cval=0, prefilter=False)
        patch = patch * rng.lognormal(-cfg["expression_log_sigma"]**2/2, cfg["expression_log_sigma"])
        y0, x0 = yi-11, xi-11
        top, left = max(0, y0), max(0, x0)
        bottom, right = min(shape[0], y0+23), min(shape[1], x0+23)
        patch = patch[top-y0:bottom-y0, left-x0:right-x0].copy()
        if patch.size == 0:
            continue
        region = (slice(top, bottom), slice(left, right))
        cells.append({"region": region, "patch": patch, "center": (y, x),
                      "template": template_id, "top": top, "left": left})
        signal[region] += patch
    # Coarse contrast matching uses fit windows only; the cells remain synthetic.
    texture = signal - ndi.gaussian_filter(signal, 4)
    variation = np.sqrt(np.maximum(ndi.gaussian_filter(texture**2, 9), 1))
    scale_map = np.clip(ref["contrast"] / variation, .2, 3.0)
    # Limit the photon budget so the diffuse background stays nonnegative.
    coarse = ndi.gaussian_filter(signal * scale_map, 8)
    budget = np.minimum(1, .85 * ref["low"] / np.maximum(coarse, 1))
    scale_map *= ndi.gaussian_filter(budget, 2)
    signal.fill(0)
    for cell in cells:
        y, x = cell["center"]
        scale = float(ndi.map_coordinates(scale_map, [[y], [x]], order=1, mode="nearest")[0])
        cell["scale"] = scale * ref["contrast_factor"]
        cell["patch"] *= cell["scale"]
        signal[cell["region"]] += cell["patch"]
    # Keep fine vessel/edge boundaries in the diffuse reference-conditioned
    # background. Cell contributions themselves are still entirely synthetic.
    cell_baseline = ndi.gaussian_filter(signal, 8)
    local_budget = np.clip(ref["low"] * ref["brightness"] / np.maximum(ndi.gaussian_filter(ref["low"] * ref["brightness"], 8), 1), 0, 1)
    for cell in cells:
        cell["patch"] *= local_budget[cell["region"]]
    signal.fill(0)
    for cell in cells:
        signal[cell["region"]] += cell["patch"]
    background = np.maximum(ref["low"] * ref["brightness"] - cell_baseline * local_budget, 0)
    clean = cfg["camera_offset_adu"] + background + signal
    ids = np.zeros(shape, np.uint16)
    winner = np.zeros(shape, np.float32)
    rows, cols, labels, strengths, metadata = [], [], [], [], []
    for k, cell in enumerate(cells, 1):
        patch = cell["patch"]
        peak = float(patch.max())
        mask = ndi.binary_fill_holes(patch >= peak * cfg["mask_relative_threshold"])
        area = int(mask.sum())
        eligible = peak >= cfg["mask_min_peak_adu"] and cfg["mask_min_area_px"] <= area <= cfg["mask_max_area_px"]
        if eligible:
            ry, rx = np.where(mask)
            rows.append((ry + cell["top"]).astype(np.uint16))
            cols.append((rx + cell["left"]).astype(np.uint16))
            labels.append(np.full(len(ry), k, np.uint16))
            strengths.append(patch[mask].astype(np.float32))
            region = cell["region"]
            take = mask & (patch > winner[region])
            ids[region][take] = k
            winner[region][take] = patch[take]
        metadata.append({"id": k, "center_yx_px": list(map(float, cell["center"])),
                         **sources[cell["template"]], "optical_scale_before_reference_envelope": cell["scale"],
                         "peak_adu": peak, "mask_area_px": area, "mask_eligible": bool(eligible)})
    if not rows:
        raise RuntimeError("No visible instance masks were generated.")
    overlaps = {"row": np.concatenate(rows), "col": np.concatenate(cols), "cell_id": np.concatenate(labels),
                "contribution_adu": np.concatenate(strengths),
                "image_shape": np.array(shape, np.uint16)}
    return clean, ids, overlaps, metadata, rng


def camera(clean, cfg, rng, repeats=16):
    signal = np.maximum(clean.astype(float) - cfg["camera_offset_adu"], 0)
    # Compound Poisson/exponential gain: E[Y]=signal, Var[Y]=shot_gain*signal.
    # Continuous gain avoids large discrete ADU steps in the dim background.
    photon_gain = cfg["shot_gain_adu"] / 2
    counts = rng.poisson(signal / photon_gain, size=(repeats, *clean.shape))
    noisy = cfg["camera_offset_adu"] + rng.gamma(counts, photon_gain)
    noisy += rng.normal(0, cfg["effective_gaussian_sigma_adu"], noisy.shape)
    return np.rint(np.clip(noisy, 0, cfg["saturation_adu"])).astype(np.uint16)


def verify_export(output, name, image, clean, ids, overlaps):
    """Independent TIFF reader plus label/overlap consistency."""
    expected = {"images": image, "clean": clean, "instance_masks": ids,
                "semantic_masks": (ids > 0).astype(np.uint8)*255}
    hashes = {}
    for folder, array in expected.items():
        path = output / folder / f"{name}.tif"
        with Image.open(path) as reader:
            actual = np.asarray(reader)
        if actual.shape != (400, 485) or actual.dtype != array.dtype or not np.array_equal(actual, array):
            raise RuntimeError(f"Independent read mismatch: {path}")
        hashes[f"{folder}/{name}.tif"] = sha256(path)
    sparse = np.load(output / "overlap_masks" / f"{name}.npz", allow_pickle=False)
    union = np.zeros(ids.shape, bool)
    union[sparse["row"], sparse["col"]] = True
    if not np.array_equal(union, ids > 0):
        raise RuntimeError("Overlap masks and instance masks have different foreground.")
    if not set(np.unique(ids)) <= set(np.r_[0, sparse["cell_id"]]):
        raise RuntimeError("Instance mask contains unknown cell IDs.")
    maximum = np.zeros(ids.shape, np.float32)
    np.maximum.at(maximum, (sparse["row"], sparse["col"]), sparse["contribution_adu"])
    take = sparse["contribution_adu"] == maximum[sparse["row"], sparse["col"]]
    expected_ids = np.full(ids.shape, 65535, np.uint16)
    np.minimum.at(expected_ids, (sparse["row"][take], sparse["col"][take]), sparse["cell_id"][take])
    expected_ids[~union] = 0
    if not np.array_equal(expected_ids, ids):
        raise RuntimeError("Instance IDs do not match the strongest contributing cell.")
    if not np.isfinite(clean).all() or image.dtype != np.uint16 or ids.dtype != np.uint16:
        raise RuntimeError("Invalid image dtype or nonfinite clean image.")
    return hashes


def comparison(output, name, real, noisy, clean, ids):
    from skimage.segmentation import find_boundaries
    def display(z):
        return np.uint8(np.clip((z.astype(float)-17800)/(32767-17800), 0, 1)*255)
    from naomi_zenodo_stage1.runtime_paths import plot_font
    panels = [real[0], noisy[0], clean]
    canvas = Image.new("RGB", (485*3+40, 450), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (panel, title) in enumerate(zip(panels, ("Real 400x485", "Synthetic 400x485", "Clean + instance masks"))):
        rgb = np.repeat(display(panel)[..., None], 3, axis=-1)
        if i == 2:
            rgb[find_boundaries(ids)] = [255, 165, 0]
        canvas.paste(Image.fromarray(rgb), (10 + i*495, 35))
        draw.text((10+i*495, 8), title, fill="black", font=plot_font(17))
    canvas.save(output / f"{name}_comparison.png")


def validation(ref, burst, clean, cfg):
    real = ref["real"]
    r, s = image_metrics(real[0]), image_metrics(burst[0])
    rc, ro = detect(real.mean(0))
    sc, so = detect(burst.mean(0))
    checks = {}
    for key in ("mean", "std", "p10", "p50", "p90", "gradient_rms"):
        offset = cfg["camera_offset_adu"] if key in ("mean", "p10", "p50", "p90") else 0
        delta = abs(s[key] - r[key]) / max(abs(r[key]-offset), 500)
        checks[key] = {"real": r[key], "synthetic": s[key], "relative_error": delta,
                       "tolerance": .25, "passed": bool(delta <= .25)}
    for key, tolerance in (("count", .25), ("diameter_median_um", .20), ("nearest_neighbor_median_um", .25)):
        delta = abs(so[key]/ro[key]-1) if ro[key] and so[key] else None
        checks[key] = {"real": ro[key], "synthetic": so[key], "relative_error": delta,
                       "tolerance": tolerance, "passed": delta is not None and delta <= tolerance}
    rn, sn = robust_noise(real), robust_noise(burst)
    checks["noise_sigma_adu"] = {"real": rn, "synthetic": sn, "relative_error": abs(sn/rn-1),
                                 "tolerance": .25, "passed": bool(abs(sn/rn-1) <= .25)}
    checks["saturation_fraction"] = {"real": r["saturation_fraction"], "synthetic": s["saturation_fraction"],
                                      "absolute_error": abs(s["saturation_fraction"]-r["saturation_fraction"]),
                                      "tolerance": .05, "passed": abs(s["saturation_fraction"]-r["saturation_fraction"]) <= .05}
    low_r = ndi.gaussian_filter(real.mean(0), 8)
    low_s = ndi.gaussian_filter(clean.astype(float), 8)
    correlation = float(np.corrcoef(low_r.ravel(), low_s.ravel())[0, 1])
    checks["large_scale_correlation"] = {"value": correlation, "minimum": .90, "passed": correlation >= .90}
    return {"similarity_checks_passed": all(c["passed"] for c in checks.values()), "checks": checks}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--count", type=int, default=4, help="Number of independent full-frame realizations (default: 4)")
    p.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    p.add_argument("--window", type=int, choices=range(8), default=5, help="Reference exposure window; 5..7 are outside the spatial fit")
    p.add_argument("--bank-seeds", type=int, nargs="+", default=[21, 22, 23], help="Cached NAOMi scenes providing individual cell shapes")
    p.add_argument("--density-factor", type=float, default=2.0, help="Simulated cells per automatically detected real cell")
    p.add_argument("--output", type=Path, help="New output directory (must not already exist)")
    args = p.parse_args()
    if args.count < 1 or args.seed < 0 or not 0 < args.density_factor <= 5:
        p.error("--count must be positive, --seed nonnegative, --density-factor in (0,5].")
    dependencies()
    global np, ndi, tifffile, Image, ImageDraw
    import numpy as np
    from scipy import ndimage as ndi
    import tifffile
    from PIL import Image, ImageDraw
    started = time.perf_counter()
    cfg = json.loads((STAGE / "parameters.json").read_text(encoding="utf-8"))
    reference = STAGE / "reference/highSNR_sample.tif"
    if not reference.is_file():
        raise FileNotFoundError(f"Clone the full repository, including {reference.relative_to(ROOT)}.")
    print("Preparing full-frame spatial template from real windows 0..4...", flush=True)
    ref = prepare_reference(reference, cfg, args.window)
    bank, sources = footprint_bank(cfg, args.bank_seeds)
    output = args.output or ROOT / "outputs" / (time.strftime("full_400x485_%Y%m%d_%H%M%S_") + str(time.time_ns() % 1000000))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for folder in ("images", "clean", "instance_masks", "semantic_masks", "overlap_masks", "cells", "diagnostics"):
        (output / folder).mkdir()
    entries, hashes = [], {}
    for index in range(args.count):
        # Index-independent RNG makes the first outputs identical when --count changes.
        seed = args.seed + index
        name = f"full_seed{seed}"
        clean, ids, overlaps, cells, rng = compose(bank, sources, cfg, ref, seed, args.density_factor)
        burst = camera(clean, cfg, rng)
        clean_export = np.minimum(clean, cfg["saturation_adu"]).astype(np.float32)
        for folder, array in (("images", burst[0]), ("clean", clean_export), ("instance_masks", ids),
                              ("semantic_masks", (ids > 0).astype(np.uint8)*255)):
            tifffile.imwrite(output / folder / f"{name}.tif", array, photometric="minisblack", metadata=None)
        np.savez_compressed(output / "overlap_masks" / f"{name}.npz", **overlaps)
        tifffile.imwrite(output / "diagnostics" / f"{name}_mean16.tif", burst.mean(0).astype(np.float32), photometric="minisblack", metadata=None)
        write_json(output / "cells" / f"{name}.json", cells)
        hashes.update(verify_export(output, name, burst[0], clean_export, ids, overlaps))
        result = validation(ref, burst, clean_export, cfg)
        entries.append({"name": name, "seed": seed, "simulated_cells": len(cells),
                        "visible_instance_count": int(np.unique(ids).size-1), **result})
        if index == 0:
            comparison(output, name, ref["real"], burst, clean_export, ids)
        failed = [k for k, v in result["checks"].items() if not v["passed"]]
        print(f"{name}: 400x485, {entries[-1]['visible_instance_count']} instance labels; "
              f"similarity: {'PASS' if not failed else 'outside targets: ' + ', '.join(failed)}", flush=True)
    report = {"export_integrity_passed": True,
              "similarity_checks_passed": all(e["similarity_checks_passed"] for e in entries),
              "shape_yx": [400, 485], "pixel_size_um": cfg["pixel_size_um"], "entries": entries,
              "independent_reader": "Pillow", "sha256": hashes,
              "limits": "Reference-conditioned 2D composition of NAOMi cell footprints, not one global 3D simulation. "
                        "Same fish supplies the spatial template and temporal check; no independent-animal validation. "
                        "Morphology checks use an automatic proxy, not expert annotations. Diagnostic mean16 is not a calcium movie."}
    write_json(output / "verification.json", report)
    write_json(output / "manifest.json", {
        "method": "Individual NAOMi optical footprint composition over one continuous full-size field",
        "shape_yx": [400, 485], "fov_yx_um": [400*.873, 485*.873], "count": args.count,
        "first_seed": args.seed, "bank_seeds": args.bank_seeds, "bank_size": len(bank),
        "density_factor": args.density_factor, "parameters": cfg,
        "camera_model": "Compound Poisson with exponential photon gain plus Gaussian read noise; photon_gain=shot_gain_adu/2. Mean signal and shot variance match the calibrated affine variance model. Excess gain factor 2 is a modeling assumption.",
        "spatial_fit_windows": [0, 1, 2, 3, 4], "comparison_window": args.window,
        "comparison_source_frame": ref["reference_frame"], "brightness_factor": ref["brightness"],
        "contrast_factor": ref["contrast_factor"], "large_scale_mean_sigma_px": 8,
        "noncell_structure_pixels": ref["structural_pixels"],
        "noncell_structure": "Connected dark cavities >=150px and elongated bright components >=80px, major axis>=30px and aspect>=3; smoothed .8px. Reference-conditioned background, not synthetic cell labels.",
        "density_and_contrast_sigma_px": 9, "minimum_projected_center_distance_px": 2,
        "reference": "https://zenodo.org/records/6293696", "reference_sample_sha256": ref["reference_sha256"],
        "script_sha256": sha256(__file__),
        "runtime_versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                             "tifffile": tifffile.__version__},
        "footprint_files_sha256": {str(s): sha256(STAGE / f"anatomy/{cfg['anatomy_tag']}_seed{s}/footprints.mat") for s in args.bank_seeds},
        "overlap_format": "Sparse COO arrays row, col, cell_id, contribution_adu; image_shape. All overlapping cell masks preserved. Instance winner independently verified from these contributions.",
        "mask_definition": "Per-cell contribution >=50% own peak; filled holes; peak>=2000ADU; area5..60px. Highest contribution wins overlaps.",
        "splitting": "All images share one real spatial template and reuse a finite NAOMi shape bank. Splitting by seed alone is insufficient for an independent fish/shape test.",
        "runtime_seconds": time.perf_counter()-started})
    print(f"DONE {output}", flush=True)
    print(f"Elapsed: {time.perf_counter()-started:.1f}s; export_integrity_passed=true; "
          f"similarity_checks_passed={report['similarity_checks_passed']}", flush=True)
    return 0 if report["similarity_checks_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Utilities for building a simple spectral library from ENVI/BSQ cubes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import spectral.io.envi as envi

from viewer import get_ignore_value, parse_wavelengths


@dataclass
class SpectralClassStats:
    class_name: str
    mean: np.ndarray
    std: np.ndarray
    count: int


def load_envi_cube(hdr_path: Path):
    """Load ENVI image and return image object, wavelengths, and ignore value."""
    img = envi.open(str(hdr_path))
    wavelengths = parse_wavelengths(img.metadata)
    if wavelengths is None:
        wavelengths = np.arange(img.nbands, dtype=float)
    ignore_value = get_ignore_value(img.metadata)
    return img, wavelengths, ignore_value


def extract_roi_spectra(
    img,
    row_min: int,
    row_max: int,
    col_min: int,
    col_max: int,
    ignore_value: float | None = None,
) -> np.ndarray:
    """Extract spectra from rectangular ROI as array (pixels, bands)."""
    row_min = max(0, int(row_min))
    col_min = max(0, int(col_min))
    row_max = min(img.nrows, int(row_max))
    col_max = min(img.ncols, int(col_max))

    if row_min >= row_max or col_min >= col_max:
        return np.empty((0, img.nbands), dtype=np.float64)

    cube = img.read_subregion((row_min, row_max), (col_min, col_max)).astype(np.float64)
    spectra = cube.reshape(-1, cube.shape[-1])

    if ignore_value is not None:
        spectra[spectra >= ignore_value] = np.nan
    spectra[spectra < 0] = np.nan

    valid = np.isfinite(spectra).any(axis=1)
    return spectra[valid]


def build_library_from_rois(hdr_path: Path, rois: dict[str, list[tuple[int, int, int, int]]]):
    """
    Build a spectral library from rectangular ROIs.

    rois format:
        {
            "water": [(r0, r1, c0, c1), ...],
            "vegetation": [(...), ...],
        }
    """
    img, wavelengths, ignore_value = load_envi_cube(hdr_path)

    stats: list[SpectralClassStats] = []
    for class_name, boxes in rois.items():
        class_spectra = []
        for row_min, row_max, col_min, col_max in boxes:
            s = extract_roi_spectra(img, row_min, row_max, col_min, col_max, ignore_value)
            if s.size > 0:
                class_spectra.append(s)

        if class_spectra:
            spectra = np.vstack(class_spectra)
            mean = np.nanmean(spectra, axis=0)
            std = np.nanstd(spectra, axis=0)
            count = int(spectra.shape[0])
        else:
            mean = np.full(img.nbands, np.nan, dtype=np.float64)
            std = np.full(img.nbands, np.nan, dtype=np.float64)
            count = 0

        stats.append(SpectralClassStats(class_name=class_name, mean=mean, std=std, count=count))

    return wavelengths, stats


def save_library_csv(output_csv: Path, wavelengths: np.ndarray, stats: list[SpectralClassStats]) -> None:
    """Save spectral means and std to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["wavelength_nm"]
    for s in stats:
        fieldnames.extend([f"{s.class_name}_mean", f"{s.class_name}_std"])

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i, wl in enumerate(wavelengths):
            row = [float(wl)]
            for s in stats:
                row.extend([
                    "" if np.isnan(s.mean[i]) else float(s.mean[i]),
                    "" if np.isnan(s.std[i]) else float(s.std[i]),
                ])
            writer.writerow(row)


def save_library_metadata(output_csv: Path, hdr_path: Path, stats: list[SpectralClassStats]) -> None:
    """Save class sample counts to CSV."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_hdr", str(hdr_path)])
        writer.writerow(["class_name", "sample_count"])
        for s in stats:
            writer.writerow([s.class_name, s.count])

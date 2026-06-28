"""Minimal Spectral Angle Mapper (SAM) implementation for Lab 5."""

from __future__ import annotations

from pathlib import Path
import csv

import numpy as np


def spectral_angle(reference: np.ndarray, sample: np.ndarray, eps: float = 1e-12) -> float:
    """Compute SAM angle in radians between two spectral vectors."""
    r = np.asarray(reference, dtype=np.float64)
    s = np.asarray(sample, dtype=np.float64)

    valid = np.isfinite(r) & np.isfinite(s)
    if valid.sum() == 0:
        return np.nan

    r = r[valid]
    s = s[valid]
    nr = np.linalg.norm(r)
    ns = np.linalg.norm(s)
    if nr < eps or ns < eps:
        return np.nan

    cosang = np.dot(r, s) / (nr * ns)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.arccos(cosang))


def classify_sam(spectra: np.ndarray, reference_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Classify each spectrum by nearest SAM reference.

    Returns:
    - labels: (n_pixels,) index of winning class
    - min_angles: (n_pixels,) minimum angle in radians
    """
    x = np.asarray(spectra, dtype=np.float64)
    refs = np.asarray(reference_matrix, dtype=np.float64)

    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    r_norm = np.linalg.norm(refs, axis=1, keepdims=True)
    denom = np.maximum(x_norm * r_norm.T, 1e-12)

    cosang = (x @ refs.T) / denom
    cosang = np.clip(cosang, -1.0, 1.0)
    angles = np.arccos(cosang)

    labels = np.argmin(angles, axis=1)
    min_angles = angles[np.arange(angles.shape[0]), labels]
    return labels, min_angles


def apply_sam_to_cube(cube_subset: np.ndarray, reference_matrix: np.ndarray):
    """
    Apply SAM to cube subset of shape (rows, cols, bands).

    Returns label map and min-angle map with shape (rows, cols).
    """
    rows, cols, bands = cube_subset.shape
    flat = cube_subset.reshape(-1, bands)
    labels, min_angles = classify_sam(flat, reference_matrix)
    return labels.reshape(rows, cols), min_angles.reshape(rows, cols)


def load_reference_from_library_csv(library_csv: Path, class_names: list[str]):
    """
    Load class mean spectra from spectral library CSV.

    Expected columns:
    - wavelength_nm
    - <class_name>_mean
    """
    wavelengths = []
    matrix = {name: [] for name in class_names}

    with library_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wavelengths.append(float(row["wavelength_nm"]))
            for name in class_names:
                key = f"{name}_mean"
                val = row.get(key, "")
                matrix[name].append(np.nan if val == "" else float(val))

    wl = np.array(wavelengths, dtype=np.float64)
    ref = np.vstack([np.array(matrix[name], dtype=np.float64) for name in class_names])
    return wl, ref


def align_reference_to_cube(
    cube_wavelengths: np.ndarray,
    ref_wavelengths: np.ndarray,
    ref_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map reference spectra to nearest cube wavelengths and return selected cube indices."""
    cube_wavelengths = np.asarray(cube_wavelengths, dtype=np.float64)
    ref_wavelengths = np.asarray(ref_wavelengths, dtype=np.float64)

    idx = np.array([int(np.argmin(np.abs(cube_wavelengths - w))) for w in ref_wavelengths], dtype=int)
    aligned = ref_matrix.copy()
    return idx, aligned

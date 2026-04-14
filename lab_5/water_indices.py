"""Water-quality index helpers for airborne hyperspectral and Sentinel-2 data."""

from __future__ import annotations

import numpy as np


def closest_band_index(wavelengths: np.ndarray, target_nm: float) -> int:
    """Return index of the wavelength closest to target."""
    wavelengths = np.asarray(wavelengths, dtype=float)
    return int(np.argmin(np.abs(wavelengths - float(target_nm))))


def _mask_invalid(arr: np.ndarray, ignore_value: float | None) -> np.ndarray:
    out = arr.astype(np.float64, copy=True)
    if ignore_value is not None:
        out[out >= ignore_value] = np.nan
    out[out < 0] = np.nan
    return out


def _norm_diff(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (a - b) / (a + b + eps)


def _resize_nearest(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize 2D array to target shape using nearest-neighbor indexing."""
    src_rows, src_cols = arr.shape
    tgt_rows, tgt_cols = target_shape
    if (src_rows, src_cols) == (tgt_rows, tgt_cols):
        return arr

    row_idx = np.clip(
        np.round(np.linspace(0, src_rows - 1, tgt_rows)).astype(int), 0, src_rows - 1
    )
    col_idx = np.clip(
        np.round(np.linspace(0, src_cols - 1, tgt_cols)).astype(int), 0, src_cols - 1
    )
    return arr[row_idx][:, col_idx]


def compute_airborne_indices(img, wavelengths: np.ndarray, ignore_value: float | None = None):
    """
    Compute simple academic proxies for water-related indices.

    Index definitions (airborne):
    - Chl-a proxy: (R705 - R665) / (R705 + R665)
    - DOC proxy:   (R620 - R560) / (R620 + R560)
    - Turbidity proxy: R665 / (R560 + eps)
    """
    i560 = closest_band_index(wavelengths, 560)
    i620 = closest_band_index(wavelengths, 620)
    i665 = closest_band_index(wavelengths, 665)
    i705 = closest_band_index(wavelengths, 705)

    b560 = _mask_invalid(img.read_band(i560), ignore_value)
    b620 = _mask_invalid(img.read_band(i620), ignore_value)
    b665 = _mask_invalid(img.read_band(i665), ignore_value)
    b705 = _mask_invalid(img.read_band(i705), ignore_value)

    chl_a = _norm_diff(b705, b665)
    doc = _norm_diff(b620, b560)
    turbidity = b665 / (b560 + 1e-8)

    return {
        "chl_a": chl_a,
        "doc": doc,
        "turbidity": turbidity,
        "band_mapping": {
            "560nm": i560,
            "620nm": i620,
            "665nm": i665,
            "705nm": i705,
        },
    }


def compute_sentinel2_indices(band_arrays: dict[str, np.ndarray]):
    """
    Compute analogous indices from Sentinel-2.

    Required keys in band_arrays (any equivalent naming accepted):
    - B03 or green (green)
    - B04 or red (red)
    - B05 or rededge1 (red edge)
    """
    def pick(*keys: str) -> np.ndarray:
        for k in keys:
            if k in band_arrays:
                return band_arrays[k].astype(np.float64)
        raise KeyError(f"Missing Sentinel-2 band. Tried keys: {keys}")

    b03 = pick("B03", "green")
    b04 = pick("B04", "red")
    b05 = pick("B05", "rededge1")

    # Sentinel-2 bands can have mixed native resolutions (10 m vs 20 m).
    target_shape = b04.shape
    b03 = _resize_nearest(b03, target_shape)
    b05 = _resize_nearest(b05, target_shape)

    chl_a = _norm_diff(b05, b04)
    doc = _norm_diff(b04, b03)
    turbidity = b04 / (b03 + 1e-8)

    return {
        "chl_a": chl_a,
        "doc": doc,
        "turbidity": turbidity,
    }

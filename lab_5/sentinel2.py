"""Minimal Sentinel-2 search/download helpers via STAC (Earth Search)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import requests
from pystac_client import Client


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1"

# Canonical names used by the analysis notebook, mapped to common STAC aliases.
ASSET_ALIASES = {
    "B03": ["B03", "green", "green-jp2"],
    "B04": ["B04", "red", "red-jp2"],
    "B05": ["B05", "rededge1", "rededge1-jp2"],
    "B11": ["B11", "swir16", "swir16-jp2"],
    "SCL": ["SCL", "scl", "scl-jp2"],
}


@dataclass
class SentinelAsset:
    item_id: str
    datetime_utc: str
    cloud_cover: float | None
    epsg: int | None
    assets: dict[str, str]


def _date_window(target_date: str, days: int) -> str:
    d = date.fromisoformat(target_date)
    start = d - timedelta(days=int(days))
    end = d + timedelta(days=int(days))
    return f"{start.isoformat()}/{end.isoformat()}"


def search_sentinel2_l2a(
    bbox: list[float],
    target_date: str,
    days_window: int = 10,
    max_items: int = 3,
):
    """
    Search Sentinel-2 L2A items near date and bbox.

    bbox format: [min_lon, min_lat, max_lon, max_lat]
    """
    catalog = Client.open(EARTH_SEARCH_URL)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=_date_window(target_date, days_window),
        query={"eo:cloud_cover": {"lt": 40}},
        max_items=max_items,
    )

    def _pick_asset_href(item_assets: dict, aliases: list[str]) -> str | None:
        for key in aliases:
            if key in item_assets:
                return item_assets[key].href
        return None

    items = []
    for item in search.items():
        assets = {}
        for canonical, aliases in ASSET_ALIASES.items():
            href = _pick_asset_href(item.assets, aliases)
            if href is not None:
                assets[canonical] = href
        items.append(
            SentinelAsset(
                item_id=item.id,
                datetime_utc=str(item.datetime),
                cloud_cover=item.properties.get("eo:cloud_cover"),
                epsg=item.properties.get("proj:epsg"),
                assets=assets,
            )
        )

    items.sort(
        key=lambda x: (
            100.0 if x.cloud_cover is None else float(x.cloud_cover),
            abs((date.fromisoformat(target_date) - date.fromisoformat(x.datetime_utc[:10])).days),
        )
    )
    return items


def download_assets(assets: dict[str, str], output_dir: Path, timeout_s: int = 60) -> dict[str, Path]:
    """Download selected Sentinel-2 assets to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    for name, href in assets.items():
        target = output_dir / f"{name}.tif"
        with requests.get(href, stream=True, timeout=timeout_s) as r:
            r.raise_for_status()
            with target.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        downloaded[name] = target

    return downloaded

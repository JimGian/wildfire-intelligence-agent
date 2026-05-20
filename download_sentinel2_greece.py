import os
import geemap
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

REGIONS = {
    "evros":       [26.15, 41.25, 26.45, 41.55],
    "rhodes":      [27.95, 36.05, 28.15, 36.25],
    "attica":      [23.55, 38.00, 23.80, 38.20],
    "evia":        [23.30, 38.40, 23.55, 38.65],
    "peloponnese": [22.10, 37.25, 22.35, 37.50],
}

# Default date ranges used when fire_start_date is not provided (2023 season)
_DEFAULT_DATE_RANGES = {
    "pre_fire":  ("2023-06-01", "2023-07-15"),
    "during":    ("2023-07-15", "2023-09-01"),
    "post_fire": ("2023-09-01", "2023-10-31"),
}

BANDS = ["B2", "B3", "B4", "B8", "B12", "NDVI", "NBR", "NDWI"]

_ee_ready = False


def _init_ee():
    global _ee_ready
    if not _ee_ready:
        import ee
        project = os.getenv("GEE_PROJECT", "wildfiregr")
        ee.Initialize(project=project)
        _ee_ready = True


def _date_ranges_from(fire_start_date: str) -> dict:
    # Heuristic tuned for the 2023 Greek fires; longer-burning events may need wider windows.
    t0 = datetime.strptime(fire_start_date, "%Y-%m-%d")
    fmt = "%Y-%m-%d"
    return {
        "pre_fire":  ((t0 - timedelta(days=45)).strftime(fmt), t0.strftime(fmt)),
        "during":    (t0.strftime(fmt), (t0 + timedelta(days=30)).strftime(fmt)),
        "post_fire": ((t0 + timedelta(days=30)).strftime(fmt), (t0 + timedelta(days=90)).strftime(fmt)),
    }


def _mask_clouds(image):
    import ee
    scl = image.select("SCL")
    clear = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
    return image.updateMask(clear)


def _compute_indices(image):
    import ee
    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr  = image.normalizedDifference(["B8", "B12"]).rename("NBR")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return image.addBands([ndvi, nbr, ndwi])


def _get_composite(bbox, date_start, date_end, max_cloud=20):
    import ee
    west, south, east, north = bbox
    aoi = ee.Geometry.Rectangle([west, south, east, north])
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(date_start, date_end)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
        .map(_mask_clouds)
        .map(_compute_indices)
    )
    count = collection.size().getInfo()
    print(f"    {count} images found")
    if count == 0:
        return None, None
    return collection.median().clip(aoi), aoi


def download_region(
    region: str,
    fire_start_date: str = None,
    output_dir: str = "data/sentinel2",
) -> dict:
    """
    Returns {period: Path} for pre_fire, during, post_fire.

    If all three TIFFs are already on disk, skips the download entirely
    and returns the cached paths. Pass fire_start_date (YYYY-MM-DD) to
    compute date ranges around a specific fire event; omit it to use the
    default 2023 season ranges.
    """
    bbox       = REGIONS[region]
    region_dir = Path(output_dir) / region
    date_ranges = (
        _date_ranges_from(fire_start_date) if fire_start_date else _DEFAULT_DATE_RANGES
    )

    paths = {period: region_dir / f"{period}.tif" for period in date_ranges}

    if all(p.exists() for p in paths.values()):
        print(f"[{region}] All TIFFs cached — skipping download")
        return {period: path for period, path in paths.items()}

    _init_ee()
    region_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Region: {region.upper()}")
    print(f"{'='*50}")

    for period, (date_start, date_end) in date_ranges.items():
        out_path = paths[period]
        if out_path.exists():
            print(f"  {period}: cached, skipping")
            continue

        print(f"\n  {period} ({date_start} -> {date_end})")
        composite, aoi = _get_composite(bbox, date_start, date_end)
        if composite is None:
            print(f"  No images found for {period} — skipping")
            continue

        print(f"  Downloading to {out_path} ...")
        try:
            geemap.download_ee_image(
                image=composite.select(BANDS),
                filename=str(out_path),
                region=aoi,
                scale=60,
                crs="EPSG:4326",
            )
            print(f"  Saved: {out_path}")
        except Exception as e:
            print(f"  Download failed: {e}")

    return {period: path for period, path in paths.items()}


def verify_downloads(output_dir: str = "data/sentinel2"):
    import rasterio
    print(f"\n{'='*50}")
    print("Verifying downloaded files...")
    print(f"{'='*50}")
    for region in REGIONS:
        region_dir = Path(output_dir) / region
        for period in _DEFAULT_DATE_RANGES:
            fpath = region_dir / f"{period}.tif"
            if fpath.exists():
                with rasterio.open(fpath) as src:
                    print(f"OK  {region}/{period}.tif  "
                          f"({src.count} bands, {src.height}x{src.width})")
            else:
                print(f"MISSING  {region}/{period}.tif")


if __name__ == "__main__":
    for region in REGIONS:
        download_region(region)
    verify_downloads()

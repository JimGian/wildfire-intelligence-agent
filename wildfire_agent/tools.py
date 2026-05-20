import os
import sys
import math
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model import WildfireDetector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# 0-indexed band positions from visualize_sentinel.py BAND_IDX
BAND_IDX = {
    "blue": 0, "green": 1, "red": 2,
    "nir":  3, "swir":  4,
    "ndvi": 5, "nbr":   6, "ndwi": 7,
}

REGIONS = {
    "evros":       [26.15, 41.25, 26.45, 41.55],
    "rhodes":      [27.95, 36.05, 28.15, 36.25],
    "attica":      [23.55, 38.00, 23.80, 38.20],
    "evia":        [23.30, 38.40, 23.55, 38.65],
    "peloponnese": [22.10, 37.25, 22.35, 37.50],
}

SENTINEL2_DIR = PROJECT_ROOT / "data" / "sentinel2"
OUTPUTS_DIR   = PROJECT_ROOT / "outputs"

# ─── Tool schemas ────────────────────────────────────────────────────────────
#
# TOOL_SCHEMAS stays in Anthropic format (source of truth / A-B reference).
# OLLAMA_TOOL_SCHEMAS is derived at import time for the local Ollama backend.

TOOL_SCHEMAS = [
    {
        "name": "run_burn_scar_model",
        "description": (
            "Returns TWO independent signals — treat them as separate evidence, not one derived from the other.\n"
            "  fire_probability: scene-level EfficientNet-B0 binary classification score (0-1). "
            "The model was fine-tuned on Sentinel-2 RGB imagery; 1 = fire detected in scene.\n"
            "  burned_ha: dNBR-derived burn area at three USGS severity thresholds. The headline "
            "figure is moderate_0.27 (USGS low/moderate boundary). Pixel area is computed from "
            "the real rasterio transform + CRS (geographic pixels are 60 m tall x 45 m wide at "
            "Greece latitudes = 2,700 m² each).\n"
            "  scene_total_ha and burned_pct_of_scene give the denominator so you can frame the "
            "number correctly. IMPORTANT: burned_ha is the area within the analyzed scene bbox, "
            "which may be a sub-region of a larger fire footprint. Use scene_total_ha and "
            "burned_pct_of_scene for context. Do not present burned_ha as the total fire size "
            "unless the scene covers the full burn extent.\n"
            "GeoTIFFs must already be on disk at data/sentinel2/<region>/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region name: evros, rhodes, attica, evia, or peloponnese",
                    "enum": ["evros", "rhodes", "attica", "evia", "peloponnese"],
                },
            },
            "required": ["region"],
        },
    },
    {
        "name": "fetch_satellite_imagery",
        "description": (
            "Downloads Sentinel-2 GeoTIFFs for a region from Google Earth Engine. "
            "If imagery is already cached on disk, skips the download and returns "
            "the existing paths. Returns a dict with keys pre_fire, during, post_fire "
            "mapping to absolute file paths."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region name: evros, rhodes, attica, evia, or peloponnese",
                    "enum": ["evros", "rhodes", "attica", "evia", "peloponnese"],
                },
                "fire_start_date": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD) of fire start; used for date range selection",
                },
            },
            "required": ["region"],
        },
    },
    {
        "name": "query_firms_active_fires",
        "description": (
            "Cross-validate satellite-derived burn extent against independent thermal-hotspot "
            "observations. Strongly recommended whenever the report makes claims about fire size "
            "or severity. Queries NASA FIRMS VIIRS_SNPP_SP; internally chunks into 5-day API calls "
            "and deduplicates by (lat, lon, acq_date). Returns deduplicated detection_count, "
            "peak_day + count, centroid, and window_covered_days over a configurable window "
            "(default 14 days, max 30). "
            "date is the window START (use ignition date). "
            "Use window_covered_days in any report — do not imply detection_count is the lifetime "
            "total if window_covered_days < fire duration. "
            "If you do not know the ignition date, call lookup_historical_context first to find it. "
            "Do not guess ignition dates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region name: evros, rhodes, attica, evia, or peloponnese",
                    "enum": ["evros", "rhodes", "attica", "evia", "peloponnese"],
                },
                "date": {
                    "type": "string",
                    "description": "Window start date ISO (YYYY-MM-DD), e.g. ignition date",
                },
                "window_days": {
                    "type": "integer",
                    "description": "Days to look forward from date (1–30, default 14)",
                    "default": 14,
                },
            },
            "required": ["region", "date"],
        },
    },
    {
        "name": "query_weather_conditions",
        "description": (
            "Fetches daily historical weather for the region centroid from the Open-Meteo "
            "archive API (archive-api.open-meteo.com; free, no key). "
            "Returns temp_max (degrees C), wind_speed_max_kmh, relative_humidity_mean_pct, and "
            "hot_dry_windy_index (= temp_max + wind_kmh - humidity_pct). "
            "hot_dry_windy_index is a rough hot/dry/windy proxy — NOT the Canadian FWI; "
            "do not label it FWI in reports."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {
                    "type": "string",
                    "description": "Region name: evros, rhodes, attica, evia, or peloponnese",
                    "enum": ["evros", "rhodes", "attica", "evia", "peloponnese"],
                },
                "date": {
                    "type": "string",
                    "description": "ISO date (YYYY-MM-DD)",
                },
            },
            "required": ["region", "date"],
        },
    },
    {
        "name": "lookup_historical_context",
        "description": (
            "Semantic similarity search over 15 markdown summaries of notable Greek / "
            "Mediterranean wildfires (2007–2024). Returns the top-3 most relevant docs."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language query, e.g. 'large fire Evros 2023' or 'Rhodes island fire'",
                },
            },
            "required": ["query"],
        },
    },
]

def _to_ollama_format(schemas: list) -> list:
    """Convert Anthropic tool schema list to Ollama/OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]


OLLAMA_TOOL_SCHEMAS = _to_ollama_format(TOOL_SCHEMAS)


# ─── Model loader (singleton, lazy) ─────────────────────────────────────────

_model = None


def _load_model() -> WildfireDetector:
    global _model
    if _model is None:
        ckpt = PROJECT_ROOT / "best_model_sentinel.pt"
        m = WildfireDetector(num_classes=1, freeze_backbone=False)
        state = torch.load(str(ckpt), map_location=DEVICE, weights_only=True)
        m.load_state_dict(state)
        m.eval()
        m.to(DEVICE)
        _model = m
    return _model


# ─── Task 1: run_burn_scar_model ────────────────────────────────────────────

def run_burn_scar_model(region: str, **_) -> dict:
    import rasterio

    pre_tif  = SENTINEL2_DIR / region / "pre_fire.tif"
    post_tif = SENTINEL2_DIR / region / "post_fire.tif"

    if not pre_tif.exists() or not post_tif.exists():
        return {"error": f"Missing GeoTIFFs for region '{region}'. Run fetch_satellite_imagery first."}

    model = _load_model()

    # Rasterio bands are 1-indexed; our BAND_IDX is 0-indexed
    nbr_band   = BAND_IDX["nbr"]   + 1  # → 7
    red_band   = BAND_IDX["red"]   + 1  # → 3
    green_band = BAND_IDX["green"] + 1  # → 2
    blue_band  = BAND_IDX["blue"]  + 1  # → 1

    with rasterio.open(pre_tif) as src:
        pre_nbr   = src.read(nbr_band).astype(np.float32)
        transform = src.transform
        crs       = src.crs
        bounds    = src.bounds

    with rasterio.open(post_tif) as src:
        post_nbr = src.read(nbr_band).astype(np.float32)
        r = src.read(red_band).astype(np.float32)
        g = src.read(green_band).astype(np.float32)
        b = src.read(blue_band).astype(np.float32)

    # Mask nodata pixels (value == 0 in both bands)
    valid = (pre_nbr != 0) & (post_nbr != 0)
    dNBR  = np.where(valid, pre_nbr - post_nbr, np.nan)

    # Pixel area in m² from the actual rasterio transform + CRS.
    # GEE exports EPSG:4326 with equal degree-size pixels (a = |e| = 0.000539°).
    # In metric space the pixel is rectangular: 60 m tall (lat) x ~45 m wide (lon)
    # because a degree of longitude = 111320 * cos(lat) metres at Greece latitudes.
    lat_center = (bounds.top + bounds.bottom) / 2
    if crs and crs.is_geographic:
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * math.cos(math.radians(lat_center))
        pixel_h_m = abs(transform.e) * m_per_deg_lat
        pixel_w_m = abs(transform.a) * m_per_deg_lon
    else:
        pixel_h_m = abs(transform.e)
        pixel_w_m = abs(transform.a)

    pixel_area_m2  = pixel_h_m * pixel_w_m
    scene_total_ha = round(int(valid.sum()) * pixel_area_m2 / 10_000, 1)

    # USGS dNBR severity thresholds (Key & Benson 2006):
    #   0.10 → low-severity floor (captures marginal + real burns, tends to overcount)
    #   0.27 → low/moderate boundary (standard "confirmed burn" threshold)
    #   0.44 → moderate/high boundary (confident high-severity burn)
    def _burned_ha(thresh: float) -> float:
        px = int(((dNBR > thresh) & valid).sum())
        return round(px * pixel_area_m2 / 10_000, 1)

    burned_ha = {
        "low_0.10":      _burned_ha(0.10),
        "moderate_0.27": _burned_ha(0.27),
        "high_0.44":     _burned_ha(0.44),
    }

    # Save the 0.27 mask as PNG (most defensible threshold for reporting).
    mask_arr  = ((dNBR > 0.27) & valid).astype(np.uint8)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    mask_path = str(OUTPUTS_DIR / f"{region}_mask.png")
    Image.fromarray(mask_arr * 255).save(mask_path)

    # ── Independent signal: EfficientNet scene-level classifier ─────────────
    # fire_probability is a binary classification score for the post-fire RGB
    # scene. It is NOT derived from dNBR — treat it as separate evidence.
    # Preprocessing matches evaluate_model_new.py tif_to_pil + inference path.
    def _stretch(band: np.ndarray) -> np.ndarray:
        valid_px = band[band > 0]
        if len(valid_px) == 0:
            return np.zeros_like(band, dtype=np.uint8)
        lo, hi = np.percentile(valid_px, 2), np.percentile(valid_px, 98)
        return ((np.clip(band, lo, hi) - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)

    rgb_arr = np.stack([_stretch(r), _stretch(g), _stretch(b)], axis=-1)
    pil_img = Image.fromarray(rgb_arr, "RGB")

    preproc = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = preproc(pil_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        fire_prob = torch.sigmoid(model(tensor)).item()

    headline_ha = burned_ha["moderate_0.27"]
    burned_pct  = round(headline_ha / scene_total_ha * 100, 1) if scene_total_ha > 0 else 0.0

    return {
        "region":               region,
        # Independent signal 1: scene-level EfficientNet classification
        "fire_probability":     round(fire_prob, 4),
        # Independent signal 2: dNBR spatial burn mask at three USGS thresholds
        "burned_ha":            burned_ha,          # headline is moderate_0.27
        "scene_total_ha":       scene_total_ha,
        "burned_pct_of_scene":  burned_pct,         # burned_ha[0.27] / scene_total_ha
        "mask_path":            mask_path,           # saved at 0.27 threshold
    }


# ─── Task 2: fetch_satellite_imagery ────────────────────────────────────────

def fetch_satellite_imagery(region: str, fire_start_date: str = None, **_) -> dict:
    from download_sentinel2_greece import download_region

    paths = download_region(region, fire_start_date=fire_start_date)

    result = {}
    missing = []
    for period, path in paths.items():
        if path.exists():
            result[period] = str(path)
        else:
            missing.append(period)

    if missing:
        result["missing"] = missing
        result["warning"] = f"GEE download failed or returned no images for: {missing}"

    return result


# ─── Task 3: query_firms_active_fires ───────────────────────────────────────

def query_firms_active_fires(region: str, date: str, window_days: int = 14, **_) -> dict:
    import requests, csv
    from io import StringIO
    from collections import Counter
    from datetime import datetime, timedelta
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("FIRMS_API_KEY", "")
    if not api_key:
        return {"error": "FIRMS_API_KEY not set in environment"}

    window_days = min(max(1, window_days), 30)   # hard cap: 30 days

    west, south, east, north = REGIONS[region]
    # Pad by 0.2° so FIRMS hotspots just outside the Sentinel-2 download bbox
    # are still captured. The Sentinel bbox was tuned for imagery coverage, not
    # fire centroid; detections can sit a fraction of a degree outside the edge.
    pad  = 0.2
    area = f"{west-pad},{south-pad},{east+pad},{north+pad}"
    # VIIRS_SNPP_SP = Standard Processing archive; works for historical dates.
    # SP API caps at 5 days per call; we chunk internally and dedup the results.
    source = "VIIRS_SNPP_SP"

    def _fetch_chunk(chunk_start: str, days: int) -> list:
        url = (
            f"https://firms.modaps.eosdis.nasa.gov/api/area/csv"
            f"/{api_key}/{source}/{area}/{days}/{chunk_start}"
        )
        try:
            r = requests.get(url, timeout=30)
        except requests.RequestException:
            return []
        if r.status_code != 200:
            return []
        text = r.text.strip()
        # FIRMS returns a plain error string when key is bad or date out of range
        if not text.startswith("latitude"):
            return []
        return list(csv.DictReader(StringIO(text)))

    # 'date' is the window START; the FIRMS API returns data going forward
    # from that date for 'days' days. Chunk forward in 5-day steps and
    # dedup by (lat, lon, acq_date) in case any boundary rows appear twice.
    all_rows: dict = {}
    current   = datetime.strptime(date, "%Y-%m-%d")
    remaining = window_days
    while remaining > 0:
        chunk = min(remaining, 5)
        for row in _fetch_chunk(current.strftime("%Y-%m-%d"), chunk):
            key = (row["latitude"], row["longitude"], row["acq_date"])
            all_rows[key] = row
        current   += timedelta(days=chunk)
        remaining -= chunk

    rows = list(all_rows.values())

    if not rows:
        return {
            "region":               region,
            "date_start":           date,
            "window_covered_days":  window_days,
            "source":               source,
            "detection_count":      0,
            "peak_day":             None,
            "peak_day_count":       0,
            "centroid_lat":         None,
            "centroid_lon":         None,
        }

    day_counts = Counter(row["acq_date"] for row in rows)
    peak_day   = max(day_counts, key=day_counts.get)
    lats = [float(row["latitude"])  for row in rows]
    lons = [float(row["longitude"]) for row in rows]

    return {
        "region":               region,
        "date_start":           date,
        "window_covered_days":  window_days,
        "source":               source,
        "detection_count":      len(rows),
        "peak_day":             peak_day,
        "peak_day_count":       day_counts[peak_day],
        "centroid_lat":         round(sum(lats) / len(lats), 4),
        "centroid_lon":         round(sum(lons) / len(lons), 4),
    }


# ─── Task 4: query_weather_conditions ───────────────────────────────────────

def query_weather_conditions(region: str, date: str, **_) -> dict:
    import requests

    west, south, east, north = REGIONS[region]
    lat = round((south + north) / 2, 4)
    lon = round((west  + east)  / 2, 4)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": date,
        "end_date":   date,
        "daily": "temperature_2m_max,wind_speed_10m_max,relative_humidity_2m_mean",
        "timezone":   "auto",
    }

    try:
        r = requests.get(url, params=params, timeout=20)
    except requests.RequestException as e:
        return {"error": f"Open-Meteo request failed: {e}"}

    if r.status_code != 200:
        return {"error": f"Open-Meteo {r.status_code}: {r.text[:300]}"}

    data = r.json()
    daily = data.get("daily", {})

    def _first(key):
        vals = daily.get(key, [None])
        return vals[0] if vals else None

    temp_max = _first("temperature_2m_max")
    wind_max = _first("wind_speed_10m_max")
    humidity = _first("relative_humidity_2m_mean")

    if temp_max is None or wind_max is None or humidity is None:
        return {"error": "Open-Meteo returned no data for this date", "raw": daily}

    # Rough fire-weather proxy: high when it's hot, windy, and dry.
    # NOT the Canadian FWI — label accordingly in any report.
    hot_dry_windy = round(temp_max + wind_max - humidity, 1)

    return {
        "region":                     region,
        "date":                       date,
        "centroid_lat":               lat,
        "centroid_lon":               lon,
        "temp_max":                   round(float(temp_max), 1),
        "wind_speed_max_kmh":         round(float(wind_max), 1),
        "relative_humidity_mean_pct": round(float(humidity), 1),
        "hot_dry_windy_index":        hot_dry_windy,
    }


# ─── Task 5: lookup_historical_context ──────────────────────────────────────

_rag_index  = None   # np.ndarray (N, 384) float32
_rag_docs   = None   # list[dict]
_rag_model  = None   # SentenceTransformer

RAG_DIR = Path(__file__).parent / "rag"


def _load_rag():
    import json
    global _rag_index, _rag_docs, _rag_model
    if _rag_index is None:
        from sentence_transformers import SentenceTransformer
        index_path = RAG_DIR / "index.npz"
        docs_path  = RAG_DIR / "docs.json"
        if not index_path.exists() or not docs_path.exists():
            raise FileNotFoundError(
                "RAG index not found. Run: python wildfire_agent/rag/build_index.py"
            )
        _rag_index = np.load(str(index_path))["embeddings"]  # (N, 384)
        _rag_docs  = json.loads(docs_path.read_text(encoding="utf-8"))
        _rag_model = SentenceTransformer("all-MiniLM-L6-v2")


def lookup_historical_context(query: str, **_) -> dict:
    _load_rag()

    q_vec = _rag_model.encode([query], convert_to_numpy=True).astype(np.float32)[0]

    # Cosine similarity: dot(q, d) / (||q|| * ||d||)
    norms = np.linalg.norm(_rag_index, axis=1)
    q_norm = np.linalg.norm(q_vec)
    with np.errstate(divide="ignore", invalid="ignore"):
        sims = np.where(
            (norms > 0) & (q_norm > 0),
            (_rag_index @ q_vec) / (norms * q_norm),
            0.0,
        )

    top_k = min(3, len(_rag_docs))
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for i in top_idx:
        doc = _rag_docs[i]
        results.append({
            "title":        doc["title"],
            "location":     doc["location"],
            "year":         doc["year"],
            "type":         doc["type"],
            "source":       doc["source"],
            "similarity":   round(float(sims[i]), 4),
            "text_excerpt": doc["text"][:400],
        })

    return {"query": query, "results": results}


# ─── Dispatcher ─────────────────────────────────────────────────────────────

def dispatch_tool(name: str, params: dict) -> dict:
    try:
        if name == "run_burn_scar_model":
            return run_burn_scar_model(**params)
        if name == "fetch_satellite_imagery":
            return fetch_satellite_imagery(**params)
        if name == "query_firms_active_fires":
            return query_firms_active_fires(**params)
        if name == "query_weather_conditions":
            return query_weather_conditions(**params)
        if name == "lookup_historical_context":
            return lookup_historical_context(**params)
        return {"error": f"Unknown tool: {name}"}
    except TypeError as exc:
        unexpected = [k for k in params if k not in (
            "region", "date", "window_days", "query", "fire_start_date"
        )]
        return {
            "status": "error",
            "error_message": (
                f"Tool '{name}' received unexpected arguments: {unexpected}. "
                "Recovered by ignoring them."
            ),
        }

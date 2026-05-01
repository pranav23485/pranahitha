import io
import re
import time
import base64
import httpx
import numpy as np
import planetary_computer
import pystac_client
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Pranahitha Spectral Insights")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OR_MODEL = "google/gemini-2.0-flash-001"
STAC_BBOX = [78.40, 17.40, 78.50, 17.50]
STAC_DATETIME = "2024-01-01/2024-02-01"

SYSTEM_PROMPT = (
    "You are the Pranahitha Tactical AI — a specialist in satellite remote sensing, Earth observation, "
    "and geospatial analysis. You ONLY answer questions related to: satellite imagery, spectral indices "
    "(NDVI, NDWI, SWIR, etc.), land cover, vegetation health, water bodies, urban mapping, "
    "environmental monitoring, geographic regions, remote sensing methodology, and Earth observation data. "
    "RULE 1: When given an image, visually analyze it and estimate vegetation health, water presence, "
    "land use, and any anomalies. Provide estimated NDVI and NDWI scores between -1 and 1. "
    "RULE 2: You analyze any geographic region provided. Always mention the region context if known. "
    "RULE 3 (CRITICAL): If a user asks ANYTHING unrelated to geography, satellite imagery, remote sensing, "
    "land cover, environmental science, or Earth observation — politely decline and redirect. "
    "Say: 'I'm a satellite and geospatial analysis AI. I can only help with remote sensing, land cover, "
    "vegetation, water bodies, and geographic analysis. Please ask me something related to Earth observation.'"
)

OR_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
    "HTTP-Referer": "http://localhost:8000",
    "X-Title": "Pranahitha Spectral Insights",
}


def or_chat(messages: list, retries=3, delay=5) -> str:
    payload = {
        "model": OR_MODEL, "temperature": 0.0,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
    }
    for attempt in range(retries):
        if attempt > 0:
            time.sleep(delay * attempt)
        resp = httpx.post(OPENROUTER_URL, json=payload, headers=OR_HEADERS, timeout=60)
        if resp.status_code == 429 and attempt < retries - 1:
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Rate limit. Please retry in a moment.")


def or_vision(image_b64: str, text_prompt: str, retries=3, delay=5) -> str:
    payload = {
        "model": OR_MODEL, "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": text_prompt},
            ]},
        ],
    }
    for attempt in range(retries):
        if attempt > 0:
            time.sleep(delay * attempt)
        resp = httpx.post(OPENROUTER_URL, json=payload, headers=OR_HEADERS, timeout=90)
        if resp.status_code == 429 and attempt < retries - 1:
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    raise RuntimeError("Rate limit. Please retry in a moment.")


def tiff_to_display_jpeg(contents: bytes) -> Optional[str]:
    try:
        import rasterio
        from rasterio.enums import ColorInterp
        from PIL import ImageFilter, ImageEnhance

        with rasterio.open(io.BytesIO(contents)) as src:
            count = src.count
            ci = src.colorinterp

            # ── Single-band: detect index raster and colorize ──────────────
            if count == 1:
                band_data = src.read(1).astype(float)
                nodata = src.nodata
                if nodata is not None:
                    band_data[band_data == nodata] = np.nan
                valid = band_data[np.isfinite(band_data)]

                fname_hint = ""  # will be set by caller if needed
                v_min = float(np.nanmin(valid)) if valid.size > 0 else 0.0
                v_max = float(np.nanmax(valid)) if valid.size > 0 else 1.0
                is_index = (v_min >= -1.05 and v_max <= 1.05 and v_max > 0.01)

                if is_index:
                    # Colorize: apply a meaningful colormap
                    # NDWI: blue (water, positive) → white (zero) → brown (dry, negative)
                    # NDVI: red (bare, negative) → yellow (sparse) → green (dense veg, positive)
                    # We'll use a diverging green-white-blue colormap that works for both
                    norm = np.clip(band_data, -1.0, 1.0)
                    norm_01 = (norm + 1.0) / 2.0  # 0=negative, 0.5=zero, 1=positive

                    # Build RGB colormap: negative→brown/red, zero→white/grey, positive→green/blue
                    r_ch = np.where(norm_01 < 0.5,
                                    np.interp(norm_01, [0, 0.5], [180, 255]),
                                    np.interp(norm_01, [0.5, 1.0], [255, 30]))
                    g_ch = np.where(norm_01 < 0.5,
                                    np.interp(norm_01, [0, 0.5], [100, 255]),
                                    np.interp(norm_01, [0.5, 1.0], [255, 160]))
                    b_ch = np.where(norm_01 < 0.5,
                                    np.interp(norm_01, [0, 0.5], [60, 255]),
                                    np.interp(norm_01, [0.5, 1.0], [255, 50]))

                    # Mask nodata as black
                    nodata_mask = ~np.isfinite(band_data)
                    r_ch[nodata_mask] = 0
                    g_ch[nodata_mask] = 0
                    b_ch[nodata_mask] = 0

                    rgb = np.stack([r_ch.astype(np.uint8), g_ch.astype(np.uint8), b_ch.astype(np.uint8)], axis=-1)
                    img = Image.fromarray(rgb, mode="RGB")
                    img.thumbnail((1280, 1280), Image.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=92)
                    return base64.b64encode(buf.getvalue()).decode()
                else:
                    # Raw reflectance single band — render as greyscale with stretch
                    valid_pos = valid[valid > 0]
                    if valid_pos.size == 0:
                        return None
                    lo, hi = np.percentile(valid_pos, 2), np.percentile(valid_pos, 98)
                    if hi == lo:
                        return None
                    norm = np.clip((band_data - lo) / (hi - lo), 0, 1)
                    grey = (np.power(norm, 1 / 1.8) * 255).astype(np.uint8)
                    rgb = np.stack([grey, grey, grey], axis=-1)
                    img = Image.fromarray(rgb, mode="RGB")
                    img.thumbnail((1280, 1280), Image.LANCZOS)
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=92)
                    return base64.b64encode(buf.getvalue()).decode()

            # ── Multi-band: find RGB channels ──────────────────────────────
            r_idx = next((i+1 for i, c in enumerate(ci) if c == ColorInterp.red), None)
            g_idx = next((i+1 for i, c in enumerate(ci) if c == ColorInterp.green), None)
            b_idx = next((i+1 for i, c in enumerate(ci) if c == ColorInterp.blue), None)

            if r_idx is None or g_idx is None or b_idx is None:
                # Try band descriptions to find RGB or NIR bands
                descs = [str(d).lower() if d else '' for d in src.descriptions]
                def find_band(keywords):
                    for kw in keywords:
                        for i, d in enumerate(descs):
                            if kw in d:
                                return i + 1
                    return None

                r_idx = find_band(['red', 'b04', 'b4', 'band4', 'band_4']) or (1 if count >= 1 else None)
                g_idx = find_band(['green', 'b03', 'b3', 'band3', 'band_3']) or (2 if count >= 2 else r_idx)
                b_idx = find_band(['blue', 'nir', 'b08', 'b8', 'band8', 'b02', 'b2']) or (3 if count >= 3 else r_idx)

                if r_idx is None:
                    r_idx = g_idx = b_idx = 1

            def stretch(band: np.ndarray) -> np.ndarray:
                band = band.astype(float)
                valid = band[band > 0]
                if valid.size == 0:
                    return np.zeros_like(band, dtype=np.uint8)
                lo, hi = np.percentile(valid, 1), np.percentile(valid, 99)
                if hi == lo:
                    return np.zeros_like(band, dtype=np.uint8)
                norm = np.clip((band - lo) / (hi - lo), 0, 1)
                gamma = np.power(norm, 1 / 1.8)
                return (gamma * 255).astype(np.uint8)

            r = stretch(src.read(r_idx))
            g = stretch(src.read(g_idx))
            b = stretch(src.read(b_idx))

        rgb = np.stack([r, g, b], axis=-1)
        img = Image.fromarray(rgb, mode="RGB")
        img.thumbnail((1280, 1280), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=130, threshold=2))
        img = ImageEnhance.Color(img).enhance(1.35)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def extract_tiff_bounds(contents: bytes) -> Optional[dict]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
        with rasterio.open(io.BytesIO(contents)) as src:
            if src.crs:
                bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
                return {
                    "min_lon": round(bounds[0], 4),
                    "min_lat": round(bounds[1], 4),
                    "max_lon": round(bounds[2], 4),
                    "max_lat": round(bounds[3], 4),
                }
    except Exception:
        pass
    return None


def rgb_to_pseudo_scores(img_array: np.ndarray):
    r = img_array[:, :, 0].astype(float)
    g = img_array[:, :, 1].astype(float)
    b = img_array[:, :, 2].astype(float)
    denom = g + r
    pseudo_ndvi = np.where(denom == 0, 0.0, (g - r) / denom)
    pseudo_ndwi = np.where(denom == 0, 0.0, (b - r) / (b + r + 1e-6))
    return round(float(np.mean(pseudo_ndvi)), 2), round(float(np.mean(pseudo_ndwi)), 2)


def pixel_level_land_cover(valid_ndvi: np.ndarray, valid_ndwi: np.ndarray) -> dict:
    """
    Pixel-level land cover using NDVI + NDWI with scientifically grounded thresholds.

    Classification (mutually exclusive, priority order: water > veg > urban > bare):
      Water:       NDWI > 0.05  (McFeeters 1996 — slightly relaxed for pseudo-NDWI from RGB)
      Vegetation:  NDVI > 0.15  (includes sparse veg — low-density crops/scrub)
      Urban:       NDVI < 0.1  AND  NDWI < -0.1
                   (built surfaces: concrete/asphalt — low NDVI, moderately negative NDWI)
      Bare soil:   everything else (NDVI 0.0–0.15 or NDWI -0.1 to 0.05)
    """
    has_ndwi = valid_ndwi is not None and valid_ndwi.size == valid_ndvi.size

    # ── Water ──────────────────────────────────────────────────────────────
    if has_ndwi:
        water_mask = valid_ndwi > 0.05
    else:
        water_mask = np.zeros(valid_ndvi.shape, dtype=bool)

    # ── Vegetation (NDVI > 0.15 — includes sparse/moderate veg) ──────────
    veg_mask = (valid_ndvi > 0.15) & ~water_mask

    # ── Urban ──────────────────────────────────────────────────────────────
    # Built-up: NDVI < 0.1 (no green cover) AND NDWI < -0.1
    # Tighter than before to avoid misclassifying bare soil as urban
    if has_ndwi:
        urban_mask = (valid_ndvi < 0.1) & (valid_ndwi < -0.1) & ~water_mask & ~veg_mask
    else:
        # Without NDWI: very low NDVI only — conservative estimate
        urban_mask = (valid_ndvi < 0.0) & ~water_mask & ~veg_mask

    # ── Bare soil: everything else ─────────────────────────────────────────
    bare_mask = ~water_mask & ~veg_mask & ~urban_mask

    total = max(valid_ndvi.size, 1)
    water_pct = round(int(np.sum(water_mask)) / total * 100)
    veg_pct   = round(int(np.sum(veg_mask))   / total * 100)
    urban_pct = round(int(np.sum(urban_mask))  / total * 100)
    bare_pct  = round(int(np.sum(bare_mask))   / total * 100)

    # Rounding correction — adjust largest category
    total_pct = water_pct + veg_pct + urban_pct + bare_pct
    if total_pct != 100:
        result = {'vegetation': veg_pct, 'water': water_pct, 'urban': urban_pct, 'bare': bare_pct}
        largest = max(result, key=result.get)
        result[largest] = max(0, result[largest] + (100 - total_pct))
        return result

    return {"vegetation": veg_pct, "water": water_pct, "urban": urban_pct, "bare": bare_pct}


def estimate_land_cover(ndvi: float, ndwi: float) -> dict:
    """
    Mean-index fallback for when pixel arrays aren't available (3-band/1-band TIFFs).
    Uses the same threshold logic as pixel_level_land_cover but applied to mean scores.
    """
    # Water: mean NDWI > 0.1 → significant water; scale up proportionally
    if ndwi > 0.3:
        water = min(80, int(55 + (ndwi - 0.3) / 0.7 * 25))
    elif ndwi > 0.1:
        water = int(30 + (ndwi - 0.1) / 0.2 * 25)
    elif ndwi > -0.1:
        # Mixed — some water pixels likely present
        water = int(8 + (ndwi + 0.1) / 0.2 * 22)
    else:
        water = 0

    # Vegetation: mean NDVI > 0.25
    if ndvi > 0.6:
        vegetation = min(85, int(65 + (ndvi - 0.6) / 0.4 * 20))
    elif ndvi > 0.4:
        vegetation = int(45 + (ndvi - 0.4) / 0.2 * 20)
    elif ndvi > 0.25:
        vegetation = int(20 + (ndvi - 0.25) / 0.15 * 25)
    else:
        vegetation = max(0, int(ndvi * 40))  # very low veg

    vegetation = min(vegetation, 98 - water)
    remaining = 100 - vegetation - water

    # Urban vs bare: urban = low NDVI + negative NDWI; bare = low NDVI + near-zero NDWI
    if ndvi < 0.15 and ndwi < -0.1:
        # Classic urban signature
        urban = min(remaining, int(remaining * 0.65))
        bare = remaining - urban
    elif ndvi < 0.25 and ndwi < 0.0:
        # Mixed urban/bare
        urban = min(remaining, int(remaining * 0.40))
        bare = remaining - urban
    else:
        # Mostly bare/transitional
        urban = min(remaining, int(remaining * 0.20))
        bare = remaining - urban

    urban = max(0, urban)
    bare = max(0, bare)
    total = vegetation + water + urban + bare
    if total != 100:
        vegetation = max(0, vegetation + (100 - total))

    return {"vegetation": vegetation, "water": water, "urban": urban, "bare": bare}

def generate_land_summary(lc: dict, ndvi: float, ndwi: float) -> dict:
    veg = lc.get("vegetation", 0)
    water = lc.get("water", 0)
    urban = lc.get("urban", 0)
    bare = lc.get("bare", 0)

    # Use land cover percentages (pixel counts) as primary signal, not mean indices
    dominant = max(lc, key=lc.get)

    if water >= 35:
        geo_type = "Aquatic / Water Body"
        geo_state = "Reservoir / lake / river — significant open water" if water >= 50 else "Mixed water-land — partial water body or wetland"
    elif water >= 15 and veg >= 20:
        geo_type = "Mixed — Riparian / Wetland Zone"
        geo_state = "Water body with surrounding vegetation — river corridor or reservoir fringe"
    elif veg >= 50:
        geo_type = "Terrestrial — Vegetated Landscape"
        geo_state = "Lush green forest" if ndvi > 0.5 else ("Moderate vegetation / mixed land" if ndvi > 0.2 else "Sparse or stressed vegetation")
    elif urban >= 40:
        geo_type = "Urban / Built-up Landscape"
        geo_state = "Dense urban area with limited green cover"
    elif bare >= 40:
        geo_type = "Arid / Bare Terrain"
        geo_state = "Dry riverbed or barren land — possible drought stress"
    elif water >= 10 and veg >= 15:
        geo_type = "Mixed — Agricultural / Riparian Zone"
        geo_state = "Farmland or mixed terrain with water features"
    else:
        geo_type = "Mixed Landscape"
        geo_state = "Heterogeneous land cover — multiple terrain types present"

    summary = (
        f"Geography Type: {geo_type}. "
        f"Current State: {geo_state}. "
        f"Cover breakdown — Vegetation: {veg}%, Water: {water}%, Urban: {urban}%, Bare Soil: {bare}%. "
        f"NDVI {ndvi} indicates {'healthy dense vegetation' if ndvi > 0.5 else ('moderate greenery' if ndvi > 0.2 else 'low vegetation or stressed land')}. "
        f"NDWI {ndwi} suggests {'significant water presence' if ndwi > 0.1 else ('mixed moisture — water pixels present despite low mean' if water > 20 else ('minimal surface water' if ndwi < 0 else 'moderate moisture'))}."
    )
    disclaimer = (
        "⚠ DISCLAIMER: This is a preliminary AI-assisted analysis based on spectral indices and visual estimation. "
        "Results are indicative only. Conduct in-situ field verification and expert review before any operational or policy decision."
    )
    return {"summary": summary, "disclaimer": disclaimer, "geo_type": geo_type, "geo_state": geo_state}


class ChatRequest(BaseModel):
    message: str
    ndvi_score: Optional[float] = None
    ndwi_score: Optional[float] = None


class AutoFetchRequest(BaseModel):
    place: str
    date_from: str
    date_to: str
    cloud_cover: Optional[int] = 15


# ── Query validation helper ────────────────────────────────────────────────
def classify_query(place: str) -> str:
    """
    Returns: 'out_of_context' | 'vague_geo' | 'ok'
    Uses AI to classify the query intent.
    """
    prompt = (
        f"The user typed this into a satellite remote-sensing analysis tool: \"{place}\"\n\n"
        "Classify this query into exactly ONE of these categories:\n"
        "1. OUT_OF_CONTEXT — clearly not related to geography, places, or Earth observation "
        "(e.g. 'which pokemon is best', 'recipe for pasta', 'who is the president')\n"
        "2. VAGUE_GEO — related to geography/environment but too vague to geocode reliably "
        "(e.g. 'forest', 'river', 'analyse vegetation', 'show me water')\n"
        "3. OK — a specific enough place name or region that can be geocoded "
        "(e.g. 'Nalamala Forest Andhra Pradesh', 'Amazon rainforest Brazil', 'Hussain Sagar Hyderabad')\n\n"
        "Reply with ONLY the category word: OUT_OF_CONTEXT, VAGUE_GEO, or OK"
    )
    try:
        result = or_chat([{"role": "user", "content": prompt}]).strip().upper()
        if "OUT_OF_CONTEXT" in result:
            return "out_of_context"
        if "VAGUE_GEO" in result:
            return "vague_geo"
        return "ok"
    except Exception:
        return "ok"  # fail open — let geocoding handle it


# ── Geocoding helper ───────────────────────────────────────────────────────
def geocode_place(place: str) -> dict:
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "Pranahitha-SpectralInsights/1.0"}
    resp = httpx.get(url, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise ValueError(f"Could not geocode '{place}'. Try a more specific place name.")
    r = results[0]
    bb = r["boundingbox"]
    min_lat, max_lat = float(bb[0]), float(bb[1])
    min_lon, max_lon = float(bb[2]), float(bb[3])
    cx = (min_lon + max_lon) / 2
    cy = (min_lat + max_lat) / 2
    half = 0.075
    return {
        "display_name": r.get("display_name", place),
        "bbox": [round(cx - half, 4), round(cy - half, 4),
                 round(cx + half, 4), round(cy + half, 4)],
        "centre": [round(cy, 4), round(cx, 4)],
    }


# ── Fast rasterio COG window read — replaces odc.stac ─────────────────────
def analyse_stac_scene_fast(item, bbox: list) -> Optional[dict]:
    """
    Read Sentinel-2 bands directly from signed COG URLs using rasterio window reads.
    Much faster than odc.stac — typically 3-8 seconds per scene vs 60+ seconds.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds
        from rasterio.enums import Resampling
        from PIL import ImageFilter, ImageEnhance

        signed = planetary_computer.sign(item)

        # Get band URLs — B04=Red, B03=Green, B08=NIR, B02=Blue (for true-colour display)
        b04_url = signed.assets.get("B04", signed.assets.get("red", None))
        b03_url = signed.assets.get("B03", signed.assets.get("green", None))
        b08_url = signed.assets.get("B08", signed.assets.get("nir", None))
        b02_url = signed.assets.get("B02", signed.assets.get("blue", None))

        if not b04_url or not b03_url or not b08_url:
            return None

        b04_href = b04_url.href
        b03_href = b03_url.href
        b08_href = b08_url.href
        b02_href = b02_url.href if b02_url else None

        TARGET_SIZE = 256  # small enough to be fast, large enough for accurate stats

        def read_band_window(href: str) -> Optional[np.ndarray]:
            try:
                with rasterio.open(href) as src:
                    # Transform bbox from WGS84 to the band's CRS
                    src_bounds = transform_bounds("EPSG:4326", src.crs, bbox[0], bbox[1], bbox[2], bbox[3])
                    win = from_bounds(*src_bounds, transform=src.transform)
                    # Clamp window to valid data area
                    win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                    if win.width <= 0 or win.height <= 0:
                        return None
                    data = src.read(
                        1, window=win,
                        out_shape=(TARGET_SIZE, TARGET_SIZE),
                        resampling=Resampling.bilinear
                    ).astype(float)
                    return data
            except Exception:
                return None

        b04 = read_band_window(b04_href)
        b03 = read_band_window(b03_href)
        b08 = read_band_window(b08_href)
        b02 = read_band_window(b02_href) if b02_href else None

        if b04 is None or b08 is None:
            return None

        valid_mask = (b04 > 0) & (b08 > 0)
        ndvi_arr = np.where((b08 + b04) == 0, np.nan, (b08 - b04) / (b08 + b04 + 1e-6))
        if b03 is not None:
            ndwi_arr = np.where((b03 + b08) == 0, np.nan, (b03 - b08) / (b03 + b08 + 1e-6))
        else:
            ndwi_arr = np.full_like(ndvi_arr, np.nan)

        valid_ndvi = ndvi_arr[valid_mask & ~np.isnan(ndvi_arr)]
        valid_ndwi = ndwi_arr[valid_mask & ~np.isnan(ndwi_arr)] if b03 is not None else np.array([])

        if valid_ndvi.size == 0:
            return None

        ndvi_score = round(float(np.mean(valid_ndvi)), 2)
        ndwi_score = round(float(np.mean(valid_ndwi)) if valid_ndwi.size > 0 else 0.0, 2)

        ndwi_for_class = valid_ndwi if valid_ndwi.size == valid_ndvi.size else None
        land_cover = pixel_level_land_cover(valid_ndvi, ndwi_for_class)

        # Build display JPEG — true-colour (B04=R, B03=G, B02=B) so it looks like a real photo.
        # Falls back to false-colour (B08 NIR as blue) if B02 unavailable.
        def stretch(band):
            if band is None:
                return np.zeros((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)
            b = band.astype(float)
            valid = b[b > 0]
            if valid.size == 0:
                return np.zeros_like(b, dtype=np.uint8)
            lo, hi = np.percentile(valid, 1), np.percentile(valid, 99)
            if hi == lo:
                return np.zeros_like(b, dtype=np.uint8)
            norm = np.clip((b - lo) / (hi - lo), 0, 1)
            return (np.power(norm, 1 / 1.8) * 255).astype(np.uint8)

        r_s = stretch(b04)                          # Red  = B04
        g_s = stretch(b03)                          # Green = B03
        b_s = stretch(b02) if b02 is not None else stretch(b08)  # Blue = B02 (true-colour) or B08 NIR fallback

        rgb = np.stack([r_s, g_s, b_s], axis=-1)
        img = Image.fromarray(rgb, mode="RGB")
        img.thumbnail((800, 800), Image.LANCZOS)
        img = img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=2))
        img = ImageEnhance.Color(img).enhance(1.3)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=88)
        display_b64 = base64.b64encode(buf.getvalue()).decode()

        date_str = item.datetime.strftime("%Y-%m-%d") if item.datetime else "unknown"
        cloud = item.properties.get("eo:cloud_cover", "?")

        # ── AI visual correction ───────────────────────────────────────────
        # True-colour display (B04=R, B03=G, B02=B) — natural colours.
        # Use AI vision to correct land cover based on what is actually visible.
        try:
            lc_str = (f"Vegetation:{land_cover.get('vegetation',0)}% "
                      f"Water:{land_cover.get('water',0)}% "
                      f"Urban:{land_cover.get('urban',0)}% "
                      f"Bare:{land_cover.get('bare',0)}%")
            tc_note = "true-colour (B04=Red, B03=Green, B02=Blue)" if b02 is not None else "false-colour (B04=Red, B03=Green, B08 NIR=Blue)"
            colour_guide = (
                "In this true-colour image: vegetation appears GREEN, water appears DARK BLUE, "
                "urban/built-up areas appear GREY/WHITE, bare soil appears BROWN/TAN."
                if b02 is not None else
                "In this false-colour image: vegetation appears BRIGHT GREEN, water appears DARK BLUE/BLACK, "
                "urban appears PURPLE/MAGENTA/GREY, bare soil appears YELLOW/BROWN."
            )
            correction_prompt = (
                f"This is a Sentinel-2 {tc_note} satellite image. {colour_guide}\n\n"
                f"Spectral band math computed: NDVI={ndvi_score}, NDWI={ndwi_score}.\n"
                f"Current land cover estimate: {lc_str}.\n\n"
                "Based on what you VISUALLY see, output CORRECTED land cover percentages. "
                "Be realistic — urban cities should show high urban %, "
                "forests should show high vegetation %, water bodies should show high water %.\n"
                "Use EXACTLY this format (integers, must sum to 100):\n"
                "VEGETATION:xx WATER:xx URBAN:xx BARE:xx\n"
                "Then provide a 2-sentence tactical insight about this scene."
            )
            ai_corr = or_vision(display_b64, correction_prompt)
            lc_match = re.search(
                r'VEGETATION:(\d+)\s+WATER:(\d+)\s+URBAN:(\d+)\s+BARE:(\d+)',
                ai_corr, re.IGNORECASE
            )
            if lc_match:
                v, w, u, b_val = (int(lc_match.group(i)) for i in range(1, 5))
                total = v + w + u + b_val
                if total > 0:
                    corrected = {
                        "vegetation": round(v * 100 / total),
                        "water":      round(w * 100 / total),
                        "urban":      round(u * 100 / total),
                        "bare":       round(b_val * 100 / total),
                    }
                    diff = 100 - sum(corrected.values())
                    if diff != 0:
                        corrected[max(corrected, key=corrected.get)] += diff
                    # Urban floor: AI-corrected urban must be >= spectral urban
                    # (urban never shrinks — if AI gives lower, keep spectral value)
                    orig_urban = land_cover.get("urban", 0)
                    if corrected["urban"] < orig_urban:
                        shortfall = orig_urban - corrected["urban"]
                        corrected["urban"] = orig_urban
                        # Reduce bare soil first, then vegetation to compensate
                        for reduce_key in ["bare", "vegetation"]:
                            take = min(shortfall, corrected[reduce_key])
                            corrected[reduce_key] -= take
                            shortfall -= take
                            if shortfall == 0:
                                break
                    land_cover = corrected
        except Exception:
            pass  # keep spectral land_cover if AI correction fails

        return {
            "date": date_str,
            "cloud_cover": cloud,
            "ndvi_score": ndvi_score,
            "ndwi_score": ndwi_score,
            "land_cover": land_cover,
            "land_summary": generate_land_summary(land_cover, ndvi_score, ndwi_score),
            "display_image": display_b64,
        }
    except Exception:
        return None


# ── /api/analyze — legacy live Sentinel-2 STAC ────────────────────────────
@app.post("/api/analyze")
async def analyze():
    try:
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"], bbox=STAC_BBOX, datetime=STAC_DATETIME,
            query={"eo:cloud_cover": {"lt": 5}},
        )
        items = list(search.items())
        if not items:
            raise HTTPException(status_code=404, detail="No Sentinel-2 scenes found.")
        item = items[0]
        result = analyse_stac_scene_fast(item, STAC_BBOX)
        if not result:
            raise HTTPException(status_code=500, detail="Scene analysis failed.")

        prompt = (
            f"Spectral data for bbox {STAC_BBOX}:\nNDVI: {result['ndvi_score']}\nNDWI: {result['ndwi_score']}\n"
            "Provide a concise tactical insight in 2-3 sentences."
        )
        ai_insight = or_chat([{"role": "user", "content": prompt}])
        result["ai_insight"] = ai_insight
        result["sensor"] = "Sentinel-2 L2A"
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/upload — image upload with satellite validation ──────────────────
@app.post("/api/upload")
async def upload_image(
    file: UploadFile = File(...),
    persona: Optional[str] = Form(None),
    scenario: Optional[str] = Form(None),
):
    allowed = {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    ext = ("." + file.filename.rsplit(".", 1)[-1].lower()) if "." in file.filename else ""
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported type '{ext}'.")

    contents = await file.read()
    size_kb = round(len(contents) / 1024, 1)
    ndvi_score = None
    ndwi_score = None
    ai_insight = None
    land_cover = None
    display_image_b64 = None
    geo_bounds = None

    if ext in {".tif", ".tiff"}:
        display_image_b64 = tiff_to_display_jpeg(contents)
        geo_bounds = extract_tiff_bounds(contents)
        try:
            import rasterio
            with rasterio.open(io.BytesIO(contents)) as src:
                band_count = src.count

                if band_count >= 4:
                    # Try to identify correct band assignments from metadata
                    raw_bands = [src.read(i+1).astype(float) for i in range(min(band_count, 8))]

                    # Compute per-band mean (ignoring zeros/nodata)
                    band_means = []
                    for b in raw_bands:
                        valid = b[b > 0]
                        band_means.append(float(np.mean(valid)) if valid.size > 0 else 0.0)

                    # ── Step 1: Try band descriptions/names first ──────────
                    descs = [str(d).lower() if d else '' for d in src.descriptions]
                    def find_idx(keywords):
                        for kw in keywords:
                            for i, d in enumerate(descs):
                                if kw in d and i < len(raw_bands):
                                    return i
                        return None

                    red_i   = find_idx(['red', 'b04', 'b4', 'band4', 'band_4', 'b_04'])
                    green_i = find_idx(['green', 'b03', 'b3', 'band3', 'band_3', 'b_03'])
                    nir_i   = find_idx(['nir', 'b08', 'b8', 'band8', 'band_8', 'b_08', 'near'])

                    def compute_indices(b_red, b_green, b_nir):
                        valid_mask = (b_red > 0) & (b_nir > 0)
                        ndvi_a = np.where((b_nir + b_red) == 0, np.nan, (b_nir - b_red) / (b_nir + b_red + 1e-6))
                        ndwi_a = np.where((b_green + b_nir) == 0, np.nan, (b_green - b_nir) / (b_green + b_nir + 1e-6))
                        vn = ndvi_a[valid_mask & ~np.isnan(ndvi_a)]
                        vw = ndwi_a[valid_mask & ~np.isnan(ndwi_a)]
                        return vn, vw, valid_mask

                    best_vn, best_vw, best_mask = None, None, None
                    best_score = -999

                    # Try metadata-identified bands first (highest priority)
                    if red_i is not None and green_i is not None and nir_i is not None:
                        vn, vw, vm = compute_indices(raw_bands[red_i], raw_bands[green_i], raw_bands[nir_i])
                        if vn.size > 0:
                            mean_ndvi = float(np.mean(vn))
                            if -0.3 <= mean_ndvi <= 0.9:
                                best_vn, best_vw, best_mask = vn, vw, vm
                                best_score = float(np.std(vn)) * 3.0  # high priority

                    # ── Step 2: Heuristic band assignments ────────────────
                    band_assignments = []
                    nb = min(band_count, len(raw_bands))
                    if nb >= 4:
                        band_assignments.append((raw_bands[2], raw_bands[1], raw_bands[3]))  # B2,B3,B4,B8: red=idx2,green=idx1,nir=idx3
                        band_assignments.append((raw_bands[0], raw_bands[1], raw_bands[3]))  # B4,B3,B2,B8: red=idx0,green=idx1,nir=idx3
                        band_assignments.append((raw_bands[0], raw_bands[1], raw_bands[2]))  # first 3 as red,green,nir
                        band_assignments.append((raw_bands[1], raw_bands[0], raw_bands[3]))  # alt: green=idx0,red=idx1,nir=idx3
                    if nb >= 8:
                        band_assignments.append((raw_bands[3], raw_bands[2], raw_bands[7]))  # 10-band: B4,B3,B8
                    if nb >= 3:
                        band_assignments.append((raw_bands[0], raw_bands[1], raw_bands[2]))

                    for b_red, b_green, b_nir in band_assignments:
                        vn, vw, vm = compute_indices(b_red, b_green, b_nir)
                        if vn.size == 0:
                            continue
                        mean_ndvi = float(np.mean(vn))
                        std_ndvi = float(np.std(vn))
                        mean_ndwi = float(np.mean(vw)) if vw.size > 0 else 0.0
                        in_range = -0.3 <= mean_ndvi <= 0.9
                        ndwi_ok = abs(mean_ndwi) < 0.85
                        score = (std_ndvi * 2.0 if in_range else -abs(mean_ndvi)) + (0.1 if ndwi_ok else -0.5)
                        if score > best_score:
                            best_score = score
                            best_vn, best_vw = vn, vw
                            best_mask = vm

                    if best_vn is not None and best_vn.size > 0:
                        ndvi_score = round(float(np.mean(best_vn)), 2)
                        ndwi_score = round(float(np.mean(best_vw)) if best_vw is not None and best_vw.size > 0 else 0.0, 2)

                        ndwi_for_class = best_vw if (best_vw is not None and best_vw.size == best_vn.size) else None
                        land_cover = pixel_level_land_cover(best_vn, ndwi_for_class)
                    else:
                        # Fallback: use band 1 as proxy
                        b1 = raw_bands[0]
                        valid = b1[b1 > 0]
                        ndvi_score = round(float(np.mean((valid - valid.min()) / (valid.max() - valid.min() + 1e-6)) * 2 - 1) if valid.size > 0 else 0.0, 2)
                        ndwi_score = 0.0
                        land_cover = estimate_land_cover(ndvi_score, ndwi_score)

                elif band_count >= 3:
                    r = src.read(1).astype(float)
                    g = src.read(2).astype(float)
                    b = src.read(3).astype(float)
                    # For 3-band RGB TIFFs: use pixel-level classification
                    # pseudo-NDVI: (NIR-Red)/(NIR+Red) — approximate with (G-R)/(G+R)
                    # pseudo-NDWI: (G-NIR)/(G+NIR) — approximate with (B-R)/(B+R)
                    # These are rough but better than mean-only fallback
                    valid_mask = (r > 0) | (g > 0) | (b > 0)
                    denom_ndvi = g + r
                    denom_ndwi = b + r
                    pseudo_ndvi_arr = np.where(denom_ndvi == 0, 0.0, (g - r) / (denom_ndvi + 1e-6))
                    pseudo_ndwi_arr = np.where(denom_ndwi == 0, 0.0, (b - r) / (denom_ndwi + 1e-6))
                    valid_ndvi_px = pseudo_ndvi_arr[valid_mask]
                    valid_ndwi_px = pseudo_ndwi_arr[valid_mask]
                    ndvi_score = round(float(np.mean(valid_ndvi_px)), 2) if valid_ndvi_px.size > 0 else 0.0
                    ndwi_score = round(float(np.mean(valid_ndwi_px)), 2) if valid_ndwi_px.size > 0 else 0.0
                    # Use pixel-level classification for better accuracy
                    ndwi_for_class = valid_ndwi_px if valid_ndwi_px.size == valid_ndvi_px.size else None
                    land_cover = pixel_level_land_cover(valid_ndvi_px, ndwi_for_class)
                elif band_count >= 1:
                    b1 = src.read(1).astype(float)
                    valid = b1[b1 != src.nodata] if src.nodata is not None else b1.flatten()
                    valid = valid[np.isfinite(valid)]

                    # Detect if this is a pre-computed index raster (NDWI/NDVI product)
                    # Index rasters have values in [-1, 1]; raw reflectance is typically 0–10000
                    if valid.size > 0:
                        v_min, v_max = float(np.min(valid)), float(np.max(valid))
                        is_index_raster = (v_min >= -1.05 and v_max <= 1.05 and v_max > 0.01)
                    else:
                        is_index_raster = False

                    fname_lower = file.filename.lower()
                    is_ndwi_file = any(k in fname_lower for k in ['ndwi', 'water', 'mndwi'])
                    is_ndvi_file = any(k in fname_lower for k in ['ndvi', 'veg', 'vegetation'])

                    if is_index_raster:
                        # This band IS the index — use it directly
                        valid_index = valid[(valid >= -1.0) & (valid <= 1.0)]
                        if is_ndwi_file or (not is_ndvi_file and float(np.mean(valid_index)) > 0.05):
                            # Treat as NDWI raster
                            ndwi_score = round(float(np.mean(valid_index)), 2)
                            ndvi_score = round(float(np.mean(valid_index)) * -0.5, 2)  # rough inverse
                            # Use NDWI values directly for pixel classification
                            ndwi_px = valid_index
                            # Synthesise NDVI: water pixels have low NDVI
                            ndvi_px = np.where(ndwi_px > 0.05, -0.1, 0.2)
                            land_cover = pixel_level_land_cover(ndvi_px, ndwi_px)
                        else:
                            # Treat as NDVI raster
                            ndvi_score = round(float(np.mean(valid_index)), 2)
                            ndwi_score = 0.0
                            ndvi_px = valid_index
                            land_cover = pixel_level_land_cover(ndvi_px, None)
                    else:
                        # Raw reflectance single band — normalise to pseudo-index
                        v_min2 = float(np.min(valid)) if valid.size > 0 else 0.0
                        v_max2 = float(np.max(valid)) if valid.size > 0 else 1.0
                        ndvi_score = round(float(np.mean((valid - v_min2) / (v_max2 - v_min2 + 1e-6))) * 2 - 1, 2) if valid.size > 0 else 0.0
                        ndwi_score = 0.0
                        land_cover = estimate_land_cover(ndvi_score, ndwi_score)
        except Exception:
            pass

    # ── AI visual correction for TIFFs ────────────────────────────────────
    # When we have a display JPEG from a TIFF, use AI vision to sanity-check
    # and correct the land cover. Spectral band math can fail (wrong band order,
    # index rasters, single-band products). AI sees the actual visual and corrects.
    if ext in {".tif", ".tiff"} and display_image_b64 and ndvi_score is not None and land_cover is not None:
        try:
            fname_lower = file.filename.lower()
            is_ndwi_file = any(k in fname_lower for k in ['ndwi', 'water', 'mndwi'])
            is_ndvi_file = any(k in fname_lower for k in ['ndvi', 'veg', 'vegetation'])

            lc_str = (f"Vegetation:{land_cover.get('vegetation',0)}% "
                      f"Water:{land_cover.get('water',0)}% "
                      f"Urban:{land_cover.get('urban',0)}% "
                      f"Bare:{land_cover.get('bare',0)}%")

            if is_ndwi_file:
                image_type_hint = (
                    "This is a colorized NDWI (Normalized Difference Water Index) raster. "
                    "GREEN/TEAL areas = high NDWI = water/moisture present. "
                    "BROWN/RED areas = low NDWI = dry land/urban/bare. "
                    "WHITE/GREY areas = near-zero NDWI = mixed/transitional terrain. "
                )
            elif is_ndvi_file:
                image_type_hint = (
                    "This is a colorized NDVI (Normalized Difference Vegetation Index) raster. "
                    "GREEN areas = high NDVI = dense healthy vegetation. "
                    "BROWN/RED areas = low NDVI = bare soil or urban. "
                    "WHITE/GREY areas = near-zero NDVI = sparse vegetation or mixed. "
                )
            else:
                image_type_hint = (
                    "This is a satellite TIFF image (may be true-colour, false-colour, or a spectral index). "
                    "Interpret the colours based on what you see: green = vegetation, blue/dark = water, "
                    "grey/purple/magenta = urban built-up, yellow/brown/tan = bare soil. "
                )

            correction_prompt = (
                f"{image_type_hint}\n"
                f"Spectral band math computed: NDVI={ndvi_score}, NDWI={ndwi_score}.\n"
                f"Current land cover estimate: {lc_str}.\n\n"
                "Look at the image carefully. Based on the VISUAL content, output CORRECTED "
                "land cover percentages that accurately reflect what you see. "
                "Be realistic and precise — if you see mostly water/blue, give high water %; "
                "if mostly green, give high vegetation %; if mostly grey/purple, give high urban %.\n"
                "Use EXACTLY this format (integers, must sum to 100):\n"
                "VEGETATION:xx WATER:xx URBAN:xx BARE:xx\n"
                "Then provide a 2-sentence tactical insight about this scene."
            )
            ai_correction = or_vision(display_image_b64, correction_prompt)
            lc_match = re.search(
                r'VEGETATION:(\d+)\s+WATER:(\d+)\s+URBAN:(\d+)\s+BARE:(\d+)',
                ai_correction, re.IGNORECASE
            )
            if lc_match:
                v = int(lc_match.group(1))
                w = int(lc_match.group(2))
                u = int(lc_match.group(3))
                b = int(lc_match.group(4))
                total = v + w + u + b
                if total > 0:
                    land_cover = {
                        "vegetation": round(v * 100 / total),
                        "water": round(w * 100 / total),
                        "urban": round(u * 100 / total),
                        "bare": round(b * 100 / total),
                    }
                    diff = 100 - sum(land_cover.values())
                    if diff != 0:
                        largest = max(land_cover, key=land_cover.get)
                        land_cover[largest] = max(0, land_cover[largest] + diff)
            insight_text = re.sub(
                r'VEGETATION:\d+\s+WATER:\d+\s+URBAN:\d+\s+BARE:\d+\s*', '',
                ai_correction, flags=re.IGNORECASE
            ).strip()
            if insight_text and len(insight_text) > 20:
                ai_insight = insight_text
        except Exception:
            pass  # keep spectral-computed land_cover if AI correction fails

    if ndvi_score is None:
        try:
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            img.thumbnail((512, 512))
            img_array = np.array(img)
            ndvi_score, ndwi_score = rgb_to_pseudo_scores(img_array)

            # Pixel-level land cover from pseudo indices (more reliable than AI text extraction)
            r_px = img_array[:, :, 0].astype(float)
            g_px = img_array[:, :, 1].astype(float)
            b_px = img_array[:, :, 2].astype(float)
            valid_mask_px = (r_px > 0) | (g_px > 0) | (b_px > 0)
            denom_v = g_px + r_px
            denom_w = b_px + r_px
            ndvi_px = np.where(denom_v == 0, 0.0, (g_px - r_px) / (denom_v + 1e-6))
            ndwi_px = np.where(denom_w == 0, 0.0, (b_px - r_px) / (denom_w + 1e-6))
            vn_px = ndvi_px[valid_mask_px]
            vw_px = ndwi_px[valid_mask_px]
            land_cover = pixel_level_land_cover(vn_px, vw_px if vw_px.size == vn_px.size else None)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            img_b64 = base64.b64encode(buf.getvalue()).decode()

            context_hint = ""
            if persona:
                context_hint += f" Persona: {persona}."
            if scenario:
                context_hint += f" Scenario: {scenario}."

            validation_prompt = (
                "You are a satellite image validator. Examine this image carefully.\n"
                "TASK 1: Determine if this is a satellite, aerial, or geo/remote-sensing image of Earth's surface.\n"
                "ACCEPT (answer YES) if the image shows ANY of: land terrain, vegetation, water bodies, urban areas, "
                "fields, forests, rivers, coastlines, mountains, deserts, agricultural land, roads/infrastructure "
                "viewed from above or at an oblique aerial angle, or any Earth observation imagery.\n"
                "REJECT (answer NO) ONLY if the image is clearly a portrait/selfie, indoor scene, "
                "food, animal close-up, product photo, screenshot, or any image that is obviously NOT "
                "showing Earth's surface from above or at distance.\n"
                "When in doubt, answer YES.\n"
                "Answer ONLY 'YES' or 'NO' on the very first line.\n"
                "TASK 2 (only if YES): "
                f"Pseudo-NDVI: {ndvi_score}, Pseudo-NDWI: {ndwi_score}.{context_hint}\n"
                "Provide a 3-sentence tactical insight for field officers about what you observe in this image "
                "(vegetation health, water presence, land use, any anomalies).\n"
                "Do NOT output land cover percentages — just the insight text.\n"
                "If NO, respond with only the word NO."
            )

            ai_text = or_vision(img_b64, validation_prompt)
            first_line = ai_text.split('\n')[0].strip().upper()

            if first_line.startswith('NO'):
                return {
                    "rejected": True,
                    "reason": (
                        "This does not appear to be a satellite or geo image. "
                        "Please upload an image from Copernicus, Sentinel, Landsat, "
                        "or a similar Earth observation source."
                    ),
                    "filename": file.filename,
                    "size_kb": size_kb,
                    "format": ext.lstrip(".").upper(),
                }

            ai_insight = re.sub(r'^YES\s*\n?', '', ai_text, flags=re.IGNORECASE).strip()

            # Extract AI-provided NDVI/NDWI scores if present (override pseudo scores)
            ndvi_m = re.search(r'ndvi[^\d\-]*(-?\d+\.?\d*)', ai_text, re.IGNORECASE)
            ndwi_m = re.search(r'ndwi[^\d\-]*(-?\d+\.?\d*)', ai_text, re.IGNORECASE)
            if ndvi_m:
                ndvi_score = round(float(ndvi_m.group(1)), 2)
            if ndwi_m:
                ndwi_score = round(float(ndwi_m.group(1)), 2)

        except Exception as e:
            ai_insight = f"Vision analysis failed: {str(e)}"

    elif ai_insight is None:
        try:
            land_cover = estimate_land_cover(ndvi_score, ndwi_score)
            prompt = f"GeoTIFF '{file.filename}' — NDVI: {ndvi_score}, NDWI: {ndwi_score}. 2-3 sentence tactical insight."
            ai_insight = or_chat([{"role": "user", "content": prompt}])
        except Exception as e:
            ai_insight = f"Scores computed. AI unavailable: {str(e)}"

    land_summary = generate_land_summary(land_cover, ndvi_score, ndwi_score) if land_cover and ndvi_score is not None else None

    return {
        "rejected": False,
        "filename": file.filename,
        "size_kb": size_kb,
        "format": ext.lstrip(".").upper(),
        "ndvi_score": ndvi_score,
        "ndwi_score": ndwi_score,
        "ai_insight": ai_insight,
        "land_cover": land_cover,
        "land_summary": land_summary,
        "display_image": display_image_b64,
        "geo_bounds": geo_bounds,
    }


# ── /api/chat ──────────────────────────────────────────────────────────────
@app.post("/api/chat")
async def chat(req: ChatRequest):
    try:
        context = ""
        if req.ndvi_score is not None and req.ndwi_score is not None:
            context = f"[Active analysis data — NDVI: {req.ndvi_score}, NDWI: {req.ndwi_score}]\n"

        # Build a rich prompt that instructs the AI to include land cover breakdown
        # when the user asks about a specific place or region
        enhanced_message = (
            req.message + "\n\n"
            "If the user is asking about a specific geographic place, region, city, lake, forest, "
            "or any Earth location — provide a realistic land cover breakdown for that location "
            "based on your knowledge. Format it EXACTLY like this in your response:\n"
            "LAND_COVER: Vegetation:XX% | Water:XX% | Urban:XX% | Bare:XX%\n"
            "(Make sure percentages sum to 100. Be realistic — e.g. Hyderabad should have high urban %, "
            "Amazon should have high vegetation %, ocean/lake should have high water %.)\n"
            "Then provide your normal geo/spectral analysis response."
        )

        reply_raw = or_chat([{"role": "user", "content": context + enhanced_message}])

        # Parse out the LAND_COVER line if present and return it separately
        lc_data = None
        lc_match = re.search(
            r'LAND_COVER:\s*Vegetation:(\d+)%\s*\|\s*Water:(\d+)%\s*\|\s*Urban:(\d+)%\s*\|\s*Bare:(\d+)%',
            reply_raw, re.IGNORECASE
        )
        if lc_match:
            v, w, u, b_val = (int(lc_match.group(i)) for i in range(1, 5))
            total = v + w + u + b_val
            if total > 0:
                lc_data = {
                    "vegetation": round(v * 100 / total),
                    "water":      round(w * 100 / total),
                    "urban":      round(u * 100 / total),
                    "bare":       round(b_val * 100 / total),
                }
                diff = 100 - sum(lc_data.values())
                if diff != 0:
                    lc_data[max(lc_data, key=lc_data.get)] += diff
            # Remove the LAND_COVER line from the visible reply
            reply = re.sub(
                r'LAND_COVER:\s*Vegetation:\d+%\s*\|\s*Water:\d+%\s*\|\s*Urban:\d+%\s*\|\s*Bare:\d+%\s*\n?',
                '', reply_raw, flags=re.IGNORECASE
            ).strip()
        else:
            reply = reply_raw

        result = {"reply": reply}
        if lc_data:
            result["land_cover"] = lc_data
            # Generate land summary for the chatbot-provided land cover
            # Use NDVI/NDWI from context if available, else estimate from land cover
            ndvi_est = req.ndvi_score if req.ndvi_score is not None else round((lc_data["vegetation"] - 50) / 100, 2)
            ndwi_est = req.ndwi_score if req.ndwi_score is not None else round((lc_data["water"] - 30) / 100, 2)
            result["land_summary"] = generate_land_summary(lc_data, ndvi_est, ndwi_est)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/autofetch — Mode 4: geocode + Sentinel-2 multi-scene fetch ────────
@app.post("/api/autofetch")
async def autofetch(req: AutoFetchRequest):
    try:
        # 1. Validate query intent before geocoding
        intent = classify_query(req.place)
        if intent == "out_of_context":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"⚠ Out of context: '{req.place}' doesn't appear to be a geographic region. "
                    "Please enter a place name, region, forest, river, city, or any Earth location."
                )
            )
        if intent == "vague_geo":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"⚠ Query too vague: '{req.place}'. Please be more specific — "
                    "e.g. 'Nalamala Forest Andhra Pradesh', 'Amazon rainforest Brazil', "
                    "'Hussain Sagar Lake Hyderabad'."
                )
            )

        # 2. Geocode
        geo = geocode_place(req.place)
        bbox = geo["bbox"]
        datetime_range = f"{req.date_from}/{req.date_to}"

        # 3. Auto-determine scene count from date range span
        from datetime import date as dt_date
        try:
            d_from = dt_date.fromisoformat(req.date_from)
            d_to = dt_date.fromisoformat(req.date_to)
            months_span = max(1, (d_to.year - d_from.year) * 12 + (d_to.month - d_from.month))
        except Exception:
            months_span = 12
        # 1 scene per ~6 months, min 2, max 4
        auto_scenes = max(2, min(4, months_span // 6))

        # 4. Search STAC with auto-relaxing cloud cover
        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        all_items = []
        used_cloud = req.cloud_cover
        for cloud_limit in [req.cloud_cover, 25, 40, 80]:
            search = catalog.search(
                collections=["sentinel-2-l2a"],
                bbox=bbox,
                datetime=datetime_range,
                query={"eo:cloud_cover": {"lt": cloud_limit}},
                max_items=50,
            )
            all_items = list(search.items())
            if all_items:
                used_cloud = cloud_limit
                break

        if not all_items:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No Sentinel-2 scenes found for '{req.place}' between {req.date_from} and {req.date_to}. "
                    "Try a wider date range or a different region."
                )
            )

        # Sort by date manually (no sortby parameter — avoids STAC server compatibility issues)
        all_items = sorted(all_items, key=lambda x: x.datetime or time.time())

        # 5. Pick evenly-spaced scenes
        n = min(auto_scenes, len(all_items))
        if n == len(all_items):
            selected = all_items
        else:
            step = len(all_items) / n
            selected = [all_items[int(i * step)] for i in range(n)]

        # 6. Analyse each scene using fast rasterio COG reads
        scenes = []
        for item in selected:
            result = analyse_stac_scene_fast(item, bbox)
            if result:
                scenes.append(result)

        if not scenes:
            raise HTTPException(status_code=500, detail="Scene analysis failed for all fetched items.")

        # 7. AI validation + silent correction of land cover values
        # Ask AI if the computed values are realistic for this location/period.
        # If AI disagrees, silently substitute with AI-corrected values.
        scene_summary = "\n".join(
            f"  {s['date']}: NDVI={s['ndvi_score']}, NDWI={s['ndwi_score']}, "
            f"Veg={s['land_cover']['vegetation']}%, Water={s['land_cover']['water']}%, "
            f"Urban={s['land_cover']['urban']}%, Bare={s['land_cover']['bare']}%"
            for s in scenes
        )

        validation_prompt = (
            f"Region: {geo['display_name']}\n"
            f"Date range: {req.date_from} to {req.date_to}\n"
            f"Sentinel-2 spectral analysis produced these land cover values:\n{scene_summary}\n\n"
            "TASK 1: For each scene, check if the land cover values are realistic for this "
            "specific location and time period based on your knowledge. "
            "Key rules: Urban areas NEVER decrease over time in Indian cities. "
            "Water bodies should reflect known lakes/rivers in the area. "
            "Vegetation should match seasonal patterns.\n\n"
            "TASK 2: If any values are unrealistic, output CORRECTED values for each scene "
            "using EXACTLY this format (one line per scene, in the same date order):\n"
            "SCENE date VEGETATION:xx WATER:xx URBAN:xx BARE:xx\n"
            "Only output SCENE lines for scenes that need correction. "
            "If all values are realistic, output: ALL_OK\n\n"
            "TASK 3: Provide a concise 3-4 sentence tactical insight covering: "
            "overall vegetation health trend, water body changes, urban expansion if any, "
            "and any notable environmental concern."
        )
        try:
            validation_response = or_chat([{"role": "user", "content": validation_prompt}])

            # Parse and apply corrections silently
            import re as _re
            for scene in scenes:
                pattern = rf"SCENE\s+{_re.escape(scene['date'])}\s+VEGETATION:(\d+)\s+WATER:(\d+)\s+URBAN:(\d+)\s+BARE:(\d+)"
                m = _re.search(pattern, validation_response, _re.IGNORECASE)
                if m:
                    v, w, u, b = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
                    total = v + w + u + b
                    if total > 0:
                        corrected = {
                            "vegetation": round(v * 100 / total),
                            "water":      round(w * 100 / total),
                            "urban":      round(u * 100 / total),
                            "bare":       round(b * 100 / total),
                        }
                        diff = 100 - sum(corrected.values())
                        if diff != 0:
                            corrected[max(corrected, key=corrected.get)] += diff
                        # Only apply if urban didn't decrease (enforce urban monotonicity)
                        if corrected["urban"] >= scene["land_cover"]["urban"]:
                            scene["land_cover"] = corrected
                            scene["land_summary"] = generate_land_summary(
                                corrected, scene["ndvi_score"], scene["ndwi_score"]
                            )

            # Extract insight text (remove SCENE lines)
            ai_insight = _re.sub(
                r'SCENE\s+\S+\s+VEGETATION:\d+\s+WATER:\d+\s+URBAN:\d+\s+BARE:\d+\s*\n?',
                '', validation_response, flags=_re.IGNORECASE
            ).replace("ALL_OK", "").strip()
            if not ai_insight or len(ai_insight) < 20:
                ai_insight = "Analysis complete. Values validated against regional knowledge."
        except Exception:
            ai_insight = "AI insight unavailable."

        return {
            "place": req.place,
            "display_name": geo["display_name"],
            "bbox": bbox,
            "centre": geo["centre"],
            "date_range": datetime_range,
            "total_found": len(all_items),
            "scenes_analysed": len(scenes),
            "cloud_used": used_cloud,
            "scenes": scenes,
            "ai_insight": ai_insight,
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── /api/persona_special — Mode 3 special analyses ────────────────────────
@app.post("/api/persona_special")
async def persona_special(
    analysis_type: str = Form(...),
    file1: UploadFile = File(...),
    file2: Optional[UploadFile] = File(None),
):
    """
    Special pixel-level analyses for Mode 3 personas:
      camouflage   — Defence: single image, green+low-NDVI camouflage detector
      encroachment — Civic: two images, water area loss in hectares
      deforestation— Natural Resources: two images, NDVI-loss heatmap
    """
    try:
        contents1 = await file1.read()
        contents2 = await file2.read() if file2 else None

        def load_rgb(data: bytes):
            img = Image.open(io.BytesIO(data)).convert("RGB")
            img.thumbnail((512, 512), Image.LANCZOS)
            return np.array(img).astype(float), img

        def load_tiff_bands(data: bytes):
            """Return (ndvi_arr, ndwi_arr, display_img_array) from TIFF or fallback to RGB."""
            try:
                import rasterio
                with rasterio.open(io.BytesIO(data)) as src:
                    count = src.count
                    if count >= 4:
                        raw = [src.read(i+1).astype(float) for i in range(min(count, 8))]
                        # Use best band combo heuristic
                        def ndvi_ndwi(r, g, n):
                            vm = (r > 0) & (n > 0)
                            ndvi = np.where((n+r)==0, 0.0, (n-r)/(n+r+1e-6))
                            ndwi = np.where((g+n)==0, 0.0, (g-n)/(g+n+1e-6))
                            return ndvi[vm], ndwi[vm]
                        combos = [(raw[2],raw[1],raw[3]),(raw[0],raw[1],raw[3]),(raw[0],raw[1],raw[2])]
                        best_ndvi, best_ndwi, best_score = None, None, -999
                        for br,bg,bn in combos:
                            vn,vw = ndvi_ndwi(br,bg,bn)
                            if vn.size == 0: continue
                            s = float(np.std(vn)) if -0.3<=float(np.mean(vn))<=0.9 else -1
                            if s > best_score:
                                best_score, best_ndvi, best_ndwi = s, vn, vw
                        return best_ndvi, best_ndwi
                    elif count == 1:
                        b = src.read(1).astype(float)
                        flat = b.flatten()
                        flat = flat[np.isfinite(flat)]
                        if flat.size > 0 and flat.min() >= -1.05 and flat.max() <= 1.05:
                            return flat, None
                        return None, None
            except Exception:
                pass
            return None, None

        # ── CAMOUFLAGE DETECTOR ───────────────────────────────────────────
        if analysis_type == "camouflage":
            arr, img = load_rgb(contents1)
            r, g, b_ch = arr[:,:,0], arr[:,:,1], arr[:,:,2]

            # Camouflage signature: high green channel + low red/blue ratio
            # (natural green paint/foliage mimicry: G dominates, R and B suppressed)
            green_dom = g / (r + g + b_ch + 1e-6)  # green fraction 0-1
            red_sup   = 1.0 - (r / (r + g + 1e-6))  # red suppression
            blue_sup  = 1.0 - (b_ch / (b_ch + g + 1e-6))  # blue suppression

            # Pseudo-NDVI from RGB
            denom = g + r
            pseudo_ndvi = np.where(denom == 0, 0.0, (g - r) / (denom + 1e-6))

            # Camouflage score: high green dominance + low NDVI (not real vegetation)
            # Real veg: NDVI > 0.3; Camouflage: NDVI 0.05-0.25 + high green fraction
            camo_score = green_dom * red_sup * blue_sup
            camo_mask  = (green_dom > 0.38) & (pseudo_ndvi > 0.02) & (pseudo_ndvi < 0.30)

            total_px   = camo_mask.size
            camo_px    = int(np.sum(camo_mask))
            camo_pct   = round(camo_px / total_px * 100, 1)

            # Confidence tiers
            if camo_pct >= 15:
                risk = "HIGH"; risk_color = "#dc2626"; risk_bg = "#fef2f2"
            elif camo_pct >= 6:
                risk = "MODERATE"; risk_color = "#d97706"; risk_bg = "#fffbeb"
            else:
                risk = "LOW"; risk_color = "#16a34a"; risk_bg = "#f0fdf4"

            # Build heatmap overlay — red pixels where camouflage detected
            h, w = camo_mask.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            overlay[camo_mask, 0] = 220  # R
            overlay[camo_mask, 1] = 30   # G
            overlay[camo_mask, 2] = 30   # B
            overlay[camo_mask, 3] = 160  # Alpha

            # Composite onto original
            base = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
            ov   = Image.fromarray(overlay, "RGBA")
            comp = Image.alpha_composite(base, ov).convert("RGB")
            buf  = io.BytesIO()
            comp.save(buf, format="JPEG", quality=88)
            overlay_b64 = base64.b64encode(buf.getvalue()).decode()

            # Mean camo score in detected zone
            zone_score = round(float(np.mean(camo_score[camo_mask])) * 100, 1) if camo_px > 0 else 0.0

            ai_prompt = (
                f"Satellite camouflage detection analysis:\n"
                f"Camouflage probability: {camo_pct}% of pixels flagged ({risk} risk)\n"
                f"Mean camouflage signature strength: {zone_score}%\n"
                f"Method: Green-dominance + pseudo-NDVI suppression filter\n\n"
                "Provide a 2-sentence tactical assessment for a defence analyst. "
                "Mention whether this warrants field verification."
            )
            ai_text = or_chat([{"role": "user", "content": ai_prompt}])

            return {
                "type": "camouflage",
                "camo_pct": camo_pct,
                "risk": risk,
                "risk_color": risk_color,
                "risk_bg": risk_bg,
                "zone_score": zone_score,
                "overlay_image": overlay_b64,
                "ai_insight": ai_text,
            }

        # ── LAKE ENCROACHMENT ─────────────────────────────────────────────
        elif analysis_type == "encroachment":
            if not contents2:
                raise HTTPException(status_code=400, detail="Two images required for encroachment analysis.")

            arr1, img1 = load_rgb(contents1)
            arr2, img2 = load_rgb(contents2)

            def water_mask_rgb(arr):
                r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                ndwi = (b - r) / (b + r + 1e-6)
                return ndwi > 0.05, ndwi

            mask1, ndwi1 = water_mask_rgb(arr1)
            mask2, ndwi2 = water_mask_rgb(arr2)

            # Resize mask2 to match mask1 if needed
            if mask1.shape != mask2.shape:
                from PIL import Image as PILImage
                m2_img = PILImage.fromarray(mask2.astype(np.uint8) * 255)
                m2_img = m2_img.resize((mask1.shape[1], mask1.shape[0]), PILImage.NEAREST)
                mask2 = np.array(m2_img) > 127

            water1_pct = round(np.sum(mask1) / mask1.size * 100, 1)
            water2_pct = round(np.sum(mask2) / mask2.size * 100, 1)
            lost_pct   = round(water1_pct - water2_pct, 1)

            # Estimate hectares — assume ~150m x 150m scene at 10m/px resolution
            # Each pixel ≈ 100m² = 0.01 ha at 10m resolution
            px_per_ha  = 100  # 10m x 10m pixels
            lost_px    = max(0, int(np.sum(mask1) - np.sum(mask2)))
            lost_ha    = round(lost_px / px_per_ha, 1)

            # Build diff overlay: red = lost water, blue = remaining water
            h, w = mask1.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            lost_mask = mask1 & ~mask2
            remain_mask = mask1 & mask2
            overlay[lost_mask,   0] = 220; overlay[lost_mask,   3] = 180  # red = lost
            overlay[remain_mask, 2] = 200; overlay[remain_mask, 3] = 120  # blue = remaining

            base = Image.fromarray(arr1.astype(np.uint8), "RGB").convert("RGBA")
            ov   = Image.fromarray(overlay, "RGBA")
            comp = Image.alpha_composite(base, ov).convert("RGB")
            buf  = io.BytesIO()
            comp.save(buf, format="JPEG", quality=88)
            overlay_b64 = base64.b64encode(buf.getvalue()).decode()

            risk = "CRITICAL" if lost_pct > 10 else ("HIGH" if lost_pct > 5 else ("MODERATE" if lost_pct > 2 else "LOW"))
            risk_color = "#dc2626" if lost_pct > 10 else ("#d97706" if lost_pct > 5 else ("#0077b6" if lost_pct > 2 else "#16a34a"))
            risk_bg    = "#fef2f2" if lost_pct > 10 else ("#fffbeb" if lost_pct > 5 else ("#eff6ff" if lost_pct > 2 else "#f0fdf4"))

            ai_prompt = (
                f"Lake/water body encroachment analysis:\n"
                f"Image 1 (baseline): {water1_pct}% water coverage\n"
                f"Image 2 (current):  {water2_pct}% water coverage\n"
                f"Water loss: {lost_pct}% ({lost_ha} estimated hectares)\n"
                f"Encroachment risk: {risk}\n\n"
                "Provide a 2-sentence civic planning assessment. "
                "Mention regulatory implications if significant encroachment is detected."
            )
            ai_text = or_chat([{"role": "user", "content": ai_prompt}])

            return {
                "type": "encroachment",
                "water1_pct": water1_pct,
                "water2_pct": water2_pct,
                "lost_pct": lost_pct,
                "lost_ha": lost_ha,
                "risk": risk,
                "risk_color": risk_color,
                "risk_bg": risk_bg,
                "overlay_image": overlay_b64,
                "ai_insight": ai_text,
            }

        # ── DEFORESTATION HEATMAP ─────────────────────────────────────────
        elif analysis_type == "deforestation":
            if not contents2:
                raise HTTPException(status_code=400, detail="Two images required for deforestation analysis.")

            arr1, img1 = load_rgb(contents1)
            arr2, img2 = load_rgb(contents2)

            def ndvi_from_rgb(arr):
                r, g = arr[:,:,0], arr[:,:,1]
                return (g - r) / (g + r + 1e-6)

            ndvi1 = ndvi_from_rgb(arr1)
            ndvi2 = ndvi_from_rgb(arr2)

            # Resize ndvi2 to match ndvi1 if needed
            if ndvi1.shape != ndvi2.shape:
                from PIL import Image as PILImage
                n2_img = PILImage.fromarray(((ndvi2 + 1) * 127.5).astype(np.uint8))
                n2_img = n2_img.resize((ndvi1.shape[1], ndvi1.shape[0]), PILImage.BILINEAR)
                ndvi2  = np.array(n2_img).astype(float) / 127.5 - 1.0

            ndvi_loss = ndvi1 - ndvi2  # positive = vegetation lost
            defor_mask = (ndvi_loss > 0.08) & (ndvi1 > 0.1)  # was vegetated, now less so

            total_px   = defor_mask.size
            defor_px   = int(np.sum(defor_mask))
            defor_pct  = round(defor_px / total_px * 100, 1)
            px_per_ha  = 100
            defor_ha   = round(defor_px / px_per_ha, 1)

            # Heatmap: intensity proportional to NDVI loss
            h, w = ndvi_loss.shape
            overlay = np.zeros((h, w, 4), dtype=np.uint8)
            loss_norm = np.clip(ndvi_loss / 0.5, 0, 1)  # normalise loss to 0-1
            overlay[:,:,0] = (loss_norm * 220).astype(np.uint8)  # red channel
            overlay[:,:,1] = 0
            overlay[:,:,2] = 0
            overlay[:,:,3] = (defor_mask * loss_norm * 200).astype(np.uint8)  # alpha = intensity

            base = Image.fromarray(arr1.astype(np.uint8), "RGB").convert("RGBA")
            ov   = Image.fromarray(overlay, "RGBA")
            comp = Image.alpha_composite(base, ov).convert("RGB")
            buf  = io.BytesIO()
            comp.save(buf, format="JPEG", quality=88)
            overlay_b64 = base64.b64encode(buf.getvalue()).decode()

            risk = "CRITICAL" if defor_pct > 15 else ("HIGH" if defor_pct > 8 else ("MODERATE" if defor_pct > 3 else "LOW"))
            risk_color = "#dc2626" if defor_pct > 15 else ("#d97706" if defor_pct > 8 else ("#0077b6" if defor_pct > 3 else "#16a34a"))
            risk_bg    = "#fef2f2" if defor_pct > 15 else ("#fffbeb" if defor_pct > 8 else ("#eff6ff" if defor_pct > 3 else "#f0fdf4"))

            mean_loss = round(float(np.mean(ndvi_loss[defor_mask])), 3) if defor_px > 0 else 0.0

            ai_prompt = (
                f"Deforestation heatmap analysis:\n"
                f"Vegetation loss detected: {defor_pct}% of scene ({defor_ha} estimated hectares)\n"
                f"Mean NDVI loss in affected zones: {mean_loss}\n"
                f"Severity: {risk}\n\n"
                "Provide a 2-sentence environmental assessment for a natural resources analyst. "
                "Mention urgency of intervention if significant deforestation is detected."
            )
            ai_text = or_chat([{"role": "user", "content": ai_prompt}])

            return {
                "type": "deforestation",
                "defor_pct": defor_pct,
                "defor_ha": defor_ha,
                "mean_loss": mean_loss,
                "risk": risk,
                "risk_color": risk_color,
                "risk_bg": risk_bg,
                "overlay_image": overlay_b64,
                "ai_insight": ai_text,
            }

        else:
            raise HTTPException(status_code=400, detail=f"Unknown analysis type: {analysis_type}")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend (must be last)
app.mount("/", StaticFiles(directory=".", html=True), name="static")

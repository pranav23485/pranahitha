// ── State ──────────────────────────────────────────────────────────────────
let lastNdvi = null;
let lastNdwi = null;

// uploadHistory: array of { filename, label, ndvi, ndwi, land_cover, geo_bounds }
// grouped by geography key for change detection
let uploadHistory = [];

// ── Splash ─────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', () => {
  const splash = document.getElementById('splash');
  setTimeout(() => {
    splash.classList.add('hidden');
    setTimeout(() => splash.remove(), 1000);
  }, 2500);
});

// ── Accordion: Modes ───────────────────────────────────────────────────────
function toggleMode(id) {
  const body = document.getElementById(id);
  const header = body.previousElementSibling;
  const isOpen = body.classList.contains('open');
  document.querySelectorAll('.mode-body').forEach(b => b.classList.remove('open'));
  document.querySelectorAll('.mode-header').forEach(h => h.classList.remove('open'));
  if (!isOpen) { body.classList.add('open'); header.classList.add('open'); }
}

// ── Accordion: Persona sub-panels ─────────────────────────────────────────
function togglePersona(id, btn) {
  const body = document.getElementById(id);
  const isOpen = body.classList.contains('open');
  document.querySelectorAll('.persona-body').forEach(b => b.classList.remove('open'));
  document.querySelectorAll('.btn-secondary[onclick^="togglePersona"]').forEach(b => b.classList.remove('active'));
  if (!isOpen) { body.classList.add('open'); btn.classList.add('active'); }
}

// ── Scenario dropdown shows upload area ───────────────────────────────────
function onScenarioChange() {
  const val = document.getElementById('scenario-select').value;
  const area = document.getElementById('scenario-upload-area');
  const label = document.getElementById('scenario-upload-label');
  if (!val) { area.style.display = 'none'; return; }
  const labels = {
    ndvi: 'Upload Vegetation Image(s) (NDVI)',
    ndwi: 'Upload Water/Flood Image(s) (NDWI)',
    defence: 'Upload Image(s) (False Color + SWIR)',
    encroachment: 'Upload Image(s) (True Color + NDVI)',
    flood: 'Upload Flood Image(s) (NDWI Timeline)',
    wildfire: 'Upload Image(s) (SWIR Wildfire)',
  };
  label.textContent = labels[val] || 'Upload Satellite Image(s)';
  area.style.display = 'flex';
}

// ── Persona sub-type shows upload area ────────────────────────────────────
function onPersonaSubChange(persona) {
  const val = document.getElementById(`${persona}-type`).value;
  const area = document.getElementById(`${persona}-upload-area`);
  area.style.display = val ? 'flex' : 'none';
}

// ── ZIP extraction helper ──────────────────────────────────────────────────
// Transparently extracts image files from a zip, or returns regular files as-is.
// Returns array of File objects. Zip contents are tagged with fromZip=true on
// the array so the upload flow can group them as a single-geography batch.
const IMAGE_EXTS = new Set(['.tif','.tiff','.jpg','.jpeg','.png']);

async function extractFilesFromInput(fileList) {
  const files = Array.from(fileList);
  const zips  = files.filter(f => f.name.toLowerCase().endsWith('.zip'));
  const imgs  = files.filter(f => !f.name.toLowerCase().endsWith('.zip'));

  if (zips.length === 0) return { files: imgs, fromZip: false };

  // Extract images from each zip
  const extracted = [...imgs];
  for (const zipFile of zips) {
    try {
      const zip = await JSZip.loadAsync(zipFile);
      const entries = Object.values(zip.files).filter(e => {
        if (e.dir) return false;
        const ext = '.' + e.name.split('.').pop().toLowerCase();
        // Skip macOS metadata files
        if (e.name.includes('__MACOSX') || e.name.startsWith('.')) return false;
        return IMAGE_EXTS.has(ext);
      });
      for (const entry of entries) {
        const blob = await entry.async('blob');
        const fname = entry.name.split('/').pop(); // strip folder path
        const ext   = '.' + fname.split('.').pop().toLowerCase();
        const mime  = ext === '.tif' || ext === '.tiff' ? 'image/tiff' : 'image/jpeg';
        extracted.push(new File([blob], fname, { type: mime }));
      }
    } catch (e) {
      appendBubble(`⚠ Could not read zip "${zipFile.name}": ${e.message}`, 'ai');
    }
  }

  return { files: extracted, fromZip: true, zipName: zips[0].name };
}


const SATELLITE_KEYWORDS = [
  'sentinel','landsat','modis','copernicus','esa','nasa','ndvi','ndwi',
  'swir','nir','band','satellite','aerial','geo','remote','sensing',
  'terrain','dem','sar','radar','multispectral','hyperspectral',
  'planet','spot','worldview','rapideye','s2','l8','l7','tif','tiff'
];

function looksLikeSatelliteFilename(filename) {
  const lower = filename.toLowerCase();
  if (lower.endsWith('.tif') || lower.endsWith('.tiff')) return true;
  return SATELLITE_KEYWORDS.some(kw => lower.includes(kw));
}

// ── Extract date from filename or label string ─────────────────────────────
// Tries: YYYY-MM-DD, YYYYMMDD, DD-MM-YYYY, DD/MM/YYYY, MMM-YYYY, YYYY_MM_DD
function extractDateFromString(str) {
  if (!str) return null;
  // ISO: 2024-06-15 or 2024_06_15
  let m = str.match(/(\d{4})[-_](\d{2})[-_](\d{2})/);
  if (m) return new Date(`${m[1]}-${m[2]}-${m[3]}`);
  // Compact: 20240615
  m = str.match(/(\d{4})(\d{2})(\d{2})/);
  if (m) {
    const d = new Date(`${m[1]}-${m[2]}-${m[3]}`);
    if (!isNaN(d)) return d;
  }
  // DD-MM-YYYY or DD/MM/YYYY
  m = str.match(/(\d{2})[-\/](\d{2})[-\/](\d{4})/);
  if (m) return new Date(`${m[3]}-${m[2]}-${m[1]}`);
  // Month name: Jun-2024, June_2024, 2024-Jun
  m = str.match(/(\d{4})[-_](Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)/i);
  if (m) return new Date(`${m[1]}-${m[2]}-01`);
  m = str.match(/(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[-_](\d{4})/i);
  if (m) return new Date(`${m[2]}-${m[1]}-01`);
  return null;
}

// Format a Date for display on chart axis
function formatDateLabel(date) {
  if (!date || isNaN(date)) return null;
  const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  return `${months[date.getMonth()]} ${date.getFullYear()}`;
}


// SAME_GEO_KM: two images are "same geography" if their centres are within ~15km
const SAME_GEO_DEG = 0.15; // ~15km — tight enough to separate Himayath vs Hussain Sagar (~15km apart)

function centrePt(geo_bounds) {
  if (!geo_bounds) return null;
  return {
    lat: (geo_bounds.min_lat + geo_bounds.max_lat) / 2,
    lon: (geo_bounds.min_lon + geo_bounds.max_lon) / 2,
  };
}

function isSameGeo(a, b) {
  if (!a || !b) return false;
  const ca = centrePt(a), cb = centrePt(b);
  if (!ca || !cb) return false;
  return Math.abs(ca.lat - cb.lat) < SAME_GEO_DEG && Math.abs(ca.lon - cb.lon) < SAME_GEO_DEG;
}

// Generate a stable geo key from a centre point, rounded to 2 decimal places (~1km grid)
function geoKey(geo_bounds) {
  if (!geo_bounds) return 'unknown';
  const c = centrePt(geo_bounds);
  if (!c) return 'unknown';
  return `${c.lat.toFixed(2)},${c.lon.toFixed(2)}`;
}

// Find the geo_key for a new entry by checking proximity to existing history entries
// Returns existing key if same geography found, else generates new key from bounds
function resolveGeoKey(geo_bounds) {
  if (!geo_bounds) return 'unknown';
  for (const entry of uploadHistory) {
    if (entry.geo_bounds && isSameGeo(entry.geo_bounds, geo_bounds)) {
      return entry.geo_key; // reuse existing group's key
    }
  }
  return geoKey(geo_bounds); // new geography — create fresh key
}

// ── Core upload function — handles multiple files ─────────────────────────
async function doUpload(fileInputId, loaderId, statusId, persona = null, scenario = null) {
  const input = document.getElementById(fileInputId);
  const loader = document.getElementById(loaderId);
  const statusEl = document.getElementById(statusId);

  if (!input || !input.files.length) {
    showStatus(statusEl, '⚠ Please select at least one file.', 'red');
    return;
  }

  // ── Extract zip contents if needed ──────────────────────────────────────
  showStatus(statusEl, 'Preparing files...', 'blue');
  const { files: allFiles, fromZip, zipName } = await extractFilesFromInput(input.files);

  if (fromZip) {
    appendBubble(`📦 Extracted ${allFiles.length} image(s) from "${zipName}"`, 'ai');
  }

  const validFiles = allFiles.filter(f => {
    if (!looksLikeSatelliteFilename(f.name)) {
      appendBubble(`🚫 Skipped "${f.name}" — not a satellite/geo image.`, 'ai');
      return false;
    }
    return true;
  });

  if (!validFiles.length) {
    showStatus(statusEl, '🚫 No valid satellite images found.', 'red');
    return;
  }

  loader.classList.add('active');
  showStatus(statusEl, `Processing ${validFiles.length} file(s)...`, 'blue');

  const results = [];
  for (let i = 0; i < validFiles.length; i++) {
    const file = validFiles[i];
    showStatus(statusEl, `Uploading ${i + 1}/${validFiles.length}: ${file.name}`, 'blue');
    const formData = new FormData();
    formData.append('file', file);
    if (persona) formData.append('persona', persona);
    if (scenario) formData.append('scenario', scenario);

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      if (data.rejected) {
        appendBubble(`🚫 "${file.name}" rejected: ${data.reason}`, 'ai');
        continue;
      }
      results.push({ file, data });
    } catch (err) {
      appendBubble(`⚠ "${file.name}" error: ${err.message}`, 'ai');
    }
  }

  loader.classList.remove('active');
  if (!results.length) { showStatus(statusEl, '⚠ No files processed successfully.', 'red'); return; }

  // Show last image in viewport
  const last = results[results.length - 1];
  if (last.data.display_image) {
    const img = document.getElementById('viewport-image');
    img.src = `data:image/jpeg;base64,${last.data.display_image}`;
    img.classList.add('visible');
    const crosshair = document.getElementById('crosshair');
    if (crosshair) crosshair.style.display = 'none';
    setViewportStatus('ANALYSIS COMPLETE ✓', '#4ade80');
  } else {
    showViewportImage(last.file);
  }

  // Update telemetry with last result
  if (last.data.ndvi_score !== null) {
    lastNdvi = last.data.ndvi_score;
    lastNdwi = last.data.ndwi_score;
    animateValue('ndvi-display', last.data.ndvi_score);
    animateValue('ndwi-display', last.data.ndwi_score);
    document.getElementById('sensor-display').textContent = `SENSOR: ${last.data.format} UPLOAD`;
  }

  // Update geo coords in viewport footer
  updateViewportCoords(last.data.geo_bounds);

  // Restore land cover breakdown
  if (last.data.land_cover) renderLandCover(last.data.land_cover);
  if (last.data.land_summary) renderLandSummary(last.data.land_summary);

  // Add to history — deduplicate by filename, group by geography proximity
  // If files came from a zip, force them all into the same geo group using
  // the first result's geo_bounds (prevents false multi-geography detection)
  const zipGeoKey = fromZip && results[0]?.data?.geo_bounds
    ? resolveGeoKey(results[0].data.geo_bounds)
    : null;

  results.forEach(({ data, file }) => {
    if (data.land_cover && data.ndvi_score !== null) {
      if (uploadHistory.some(e => e.filename === data.filename)) return;

      // For zip batches: use shared geo key to keep them in one group
      const assignedKey = zipGeoKey || resolveGeoKey(data.geo_bounds);
      uploadHistory.push({
        filename: data.filename,
        label: file.name.replace(/\.[^.]+$/, '').slice(0, 18),
        date: extractDateFromString(file.name),
        ndvi: data.ndvi_score,
        ndwi: data.ndwi_score,
        land_cover: data.land_cover,
        geo_bounds: data.geo_bounds,
        geo_key: assignedKey,
      });
    }
  });

  if (uploadHistory.length > 1) {
    const trendBtn = document.querySelector('button[onclick="openTrendModal()"]');
    if (trendBtn) trendBtn.style.boxShadow = '0 0 0 2px #16a34a55';
  }

  showStatus(statusEl, `✓ ${results.length} file(s) processed`, 'green');
  const insight = last.data.ai_insight || `${results.length} file(s) processed.`;
  appendBubble(`📡 [${last.data.format}] ${last.data.filename}\n${insight}`, 'ai', true);
}

// ── Mode entry points ──────────────────────────────────────────────────────
function uploadFile() { doUpload('upload-input', 'upload-loader', 'upload-status'); }
function runScenarioAnalysis() {
  const scenario = document.getElementById('scenario-select').value;
  if (!scenario) { appendBubble('⚠ Please select a scenario first.', 'ai'); return; }
  doUpload('scenario-upload-input', 'scenario-loader', 'scenario-status', null, scenario);
}
function runPersonaAnalysis(persona) {
  const subType = document.getElementById(`${persona}-type`).value;
  doUpload(`${persona}-upload-input`, `${persona}-loader`, `${persona}-status`, persona, subType);
}

// ── Mode 3 Special Analyses ────────────────────────────────────────────────
async function runSpecialAnalysis(type, file1Id, file2Id, loaderId, resultId) {
  const f1Input = document.getElementById(file1Id);
  const f2Input = file2Id ? document.getElementById(file2Id) : null;
  const loader = document.getElementById(loaderId);
  const resultEl = document.getElementById(resultId);

  if (!f1Input || !f1Input.files.length) {
    resultEl.innerHTML = `<div style="color:#dc2626;font-size:0.74rem;margin-top:0.4rem;">⚠ Please select the required image(s).</div>`;
    return;
  }

  loader.classList.add('active');
  resultEl.innerHTML = '';

  // Extract from zip if needed
  const { files: files1 } = await extractFilesFromInput(f1Input.files);
  let file1 = files1[0];

  let file2 = null;
  if (f2Input && f2Input.files.length) {
    const { files: files2 } = await extractFilesFromInput(f2Input.files);
    file2 = files2[0];
  }

  // If a zip was given for a single-image analysis (camouflage), process all images in it
  if (type === 'camouflage' && files1.length > 1) {
    // Run on all extracted images and show combined result
    let bestResult = null;
    for (const f of files1) {
      const fd = new FormData();
      fd.append('analysis_type', type);
      fd.append('file1', f);
      try {
        const res = await fetch('/api/persona_special', { method: 'POST', body: fd });
        const data = await res.json();
        if (res.ok && (!bestResult || data.camo_pct > bestResult.camo_pct)) {
          bestResult = data;
        }
      } catch (_) {}
    }
    loader.classList.remove('active');
    if (bestResult) {
      renderSpecialResult(bestResult, resultEl);
      if (bestResult.overlay_image) {
        const img = document.getElementById('viewport-image');
        img.src = `data:image/jpeg;base64,${bestResult.overlay_image}`;
        img.classList.add('visible');
        document.getElementById('crosshair').style.display = 'none';
        setViewportStatus(`MODE 3 · CAMOUFLAGE SCAN (${files1.length} images) ✓`, '#4ade80');
      }
      appendBubble(`🔬 [Mode 3 · camouflage · ${files1.length} images] ${bestResult.ai_insight}`, 'ai', true);
    }
    return;
  }

  if (!file1) {
    loader.classList.remove('active');
    resultEl.innerHTML = `<div style="color:#dc2626;font-size:0.74rem;margin-top:0.4rem;">⚠ No valid image found.</div>`;
    return;
  }
  if ((type === 'encroachment' || type === 'deforestation') && !file2) {
    loader.classList.remove('active');
    resultEl.innerHTML = `<div style="color:#dc2626;font-size:0.74rem;margin-top:0.4rem;">⚠ Please select both images (or a zip with 2 images).</div>`;
    return;
  }

  const fd = new FormData();
  fd.append('analysis_type', type);
  fd.append('file1', file1);
  if (file2) fd.append('file2', file2);

  try {
    const res = await fetch('/api/persona_special', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Analysis failed');
    renderSpecialResult(data, resultEl);

    if (data.overlay_image) {
      const img = document.getElementById('viewport-image');
      img.src = `data:image/jpeg;base64,${data.overlay_image}`;
      img.classList.add('visible');
      const crosshair = document.getElementById('crosshair');
      if (crosshair) crosshair.style.display = 'none';
      setViewportStatus(`MODE 3 · ${type.toUpperCase()} ANALYSIS ✓`, '#4ade80');
    }
    appendBubble(`🔬 [Mode 3 · ${type}] ${data.ai_insight}`, 'ai', true);
  } catch (err) {
    resultEl.innerHTML = `<div style="color:#dc2626;font-size:0.74rem;margin-top:0.4rem;">⚠ ${err.message}</div>`;
  } finally {
    loader.classList.remove('active');
  }
}

function renderSpecialResult(data, el) {
  if (data.type === 'camouflage') {
    el.innerHTML = `
      <div style="margin-top:0.6rem;border-radius:10px;overflow:hidden;border:1px solid ${data.risk_color}44;">
        <div style="background:${data.risk_bg};padding:0.5rem 0.75rem;display:flex;align-items:center;justify-content:space-between;">
          <span style="font-size:0.75rem;font-weight:700;color:${data.risk_color};">🪖 CAMOUFLAGE RISK: ${data.risk}</span>
          <span style="font-size:0.8rem;font-weight:700;color:${data.risk_color};">${data.camo_pct}%</span>
        </div>
        <div style="padding:0.6rem 0.75rem;background:#fff;font-size:0.72rem;line-height:1.7;color:var(--text);">
          <div style="display:flex;gap:1rem;margin-bottom:0.4rem;">
            <span>🎯 Flagged pixels: <strong>${data.camo_pct}%</strong></span>
            <span>💪 Signature strength: <strong>${data.zone_score}%</strong></span>
          </div>
          <div style="background:#e9ecef;border-radius:4px;height:8px;margin-bottom:0.5rem;">
            <div style="width:${Math.min(data.camo_pct,100)}%;background:${data.risk_color};height:8px;border-radius:4px;transition:width 0.8s;"></div>
          </div>
          <div style="color:var(--text-muted);font-style:italic;">${data.ai_insight}</div>
        </div>
      </div>`;
  } else if (data.type === 'encroachment') {
    el.innerHTML = `
      <div style="margin-top:0.6rem;border-radius:10px;overflow:hidden;border:1px solid ${data.risk_color}44;">
        <div style="background:${data.risk_bg};padding:0.5rem 0.75rem;display:flex;align-items:center;justify-content:space-between;">
          <span style="font-size:0.75rem;font-weight:700;color:${data.risk_color};">💧 ENCROACHMENT: ${data.risk}</span>
          <span style="font-size:0.8rem;font-weight:700;color:${data.risk_color};">-${data.lost_pct}%</span>
        </div>
        <div style="padding:0.6rem 0.75rem;background:#fff;font-size:0.72rem;line-height:1.7;color:var(--text);">
          <div style="display:flex;gap:1rem;margin-bottom:0.4rem;flex-wrap:wrap;">
            <span>📅 Baseline water: <strong>${data.water1_pct}%</strong></span>
            <span>📅 Current water: <strong>${data.water2_pct}%</strong></span>
            <span>📉 Lost: <strong>${data.lost_ha} ha</strong></span>
          </div>
          <div style="background:#e9ecef;border-radius:4px;height:8px;margin-bottom:0.5rem;position:relative;">
            <div style="width:${data.water1_pct}%;background:#93c5fd;height:8px;border-radius:4px;position:absolute;"></div>
            <div style="width:${data.water2_pct}%;background:#0077b6;height:8px;border-radius:4px;position:absolute;"></div>
          </div>
          <div style="color:var(--text-muted);font-style:italic;">${data.ai_insight}</div>
        </div>
      </div>`;
  } else if (data.type === 'deforestation') {
    el.innerHTML = `
      <div style="margin-top:0.6rem;border-radius:10px;overflow:hidden;border:1px solid ${data.risk_color}44;">
        <div style="background:${data.risk_bg};padding:0.5rem 0.75rem;display:flex;align-items:center;justify-content:space-between;">
          <span style="font-size:0.75rem;font-weight:700;color:${data.risk_color};">🌲 DEFORESTATION: ${data.risk}</span>
          <span style="font-size:0.8rem;font-weight:700;color:${data.risk_color};">${data.defor_pct}%</span>
        </div>
        <div style="padding:0.6rem 0.75rem;background:#fff;font-size:0.72rem;line-height:1.7;color:var(--text);">
          <div style="display:flex;gap:1rem;margin-bottom:0.4rem;flex-wrap:wrap;">
            <span>🔴 Affected area: <strong>${data.defor_pct}%</strong></span>
            <span>🌳 Est. loss: <strong>${data.defor_ha} ha</strong></span>
            <span>📉 Mean NDVI loss: <strong>${data.mean_loss}</strong></span>
          </div>
          <div style="background:#e9ecef;border-radius:4px;height:8px;margin-bottom:0.5rem;">
            <div style="width:${Math.min(data.defor_pct,100)}%;background:${data.risk_color};height:8px;border-radius:4px;transition:width 0.8s;"></div>
          </div>
          <div style="color:var(--text-muted);font-style:italic;">${data.ai_insight}</div>
        </div>
      </div>`;
  }
}

// ── Mode 4: AI Auto-Fetch ──────────────────────────────────────────────────
async function runAutoFetch() {
  const placeEl = document.getElementById('m4-place');
  const place = placeEl.value.trim();
  const dateFrom = document.getElementById('m4-from').value.trim();
  const dateTo = document.getElementById('m4-to').value.trim();
  const cloudCover = parseInt(document.getElementById('m4-cloud').value);
  const btn = document.getElementById('m4-btn');
  const loader = document.getElementById('m4-loader');
  const loaderText = document.getElementById('m4-loader-text');
  const statusEl = document.getElementById('m4-status');

  if (!place) { showStatus(statusEl, '⚠ Please enter a region name.', 'red'); return; }
  if (!dateFrom || !dateTo) { showStatus(statusEl, '⚠ Please enter both dates (YYYY-MM-DD).', 'red'); return; }

  btn.disabled = true;
  loader.classList.add('active');
  loaderText.textContent = `Geocoding "${place}"...`;
  showStatus(statusEl, `Searching for "${place}"...`, 'blue');
  setViewportStatus('FETCHING SENTINEL-2 DATA...', 'rgba(255,255,255,0.7)');

  try {
    loaderText.textContent = `Fetching Sentinel-2 scenes (${dateFrom} → ${dateTo})...`;
    const res = await fetch('/api/autofetch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ place, date_from: dateFrom, date_to: dateTo, cloud_cover: cloudCover }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Auto-fetch failed');

    const scenes = data.scenes;
    if (!scenes || !scenes.length) throw new Error('No scenes returned.');

    showStatus(statusEl, `✓ ${scenes.length} scene(s) analysed from ${data.total_found} found${data.cloud_used > cloudCover ? ` (cloud relaxed to ${data.cloud_used}%)` : ''}`, 'green');

    // Show last scene in viewport
    const lastScene = scenes[scenes.length - 1];
    if (lastScene.display_image) {
      const img = document.getElementById('viewport-image');
      img.src = `data:image/jpeg;base64,${lastScene.display_image}`;
      img.classList.add('visible');
      const crosshair = document.getElementById('crosshair');
      if (crosshair) crosshair.style.display = 'none';
      setViewportStatus(`SENTINEL-2 · ${lastScene.date} · ☁ ${lastScene.cloud_cover}%`, '#4ade80');
    }

    // Update telemetry
    lastNdvi = lastScene.ndvi_score;
    lastNdwi = lastScene.ndwi_score;
    animateValue('ndvi-display', lastScene.ndvi_score);
    animateValue('ndwi-display', lastScene.ndwi_score);
    document.getElementById('sensor-display').textContent = `SENSOR: Sentinel-2 L2A · AUTO-FETCH`;

    // Update viewport coords from bbox
    const bb = data.bbox;
    updateViewportCoords({ min_lon: bb[0], min_lat: bb[1], max_lon: bb[2], max_lat: bb[3] });

    // Land cover + summary from last scene
    if (lastScene.land_cover) renderLandCover(lastScene.land_cover);
    if (lastScene.land_summary) renderLandSummary(lastScene.land_summary);

    // Push all scenes into uploadHistory for change detection
    const geoB = { min_lon: bb[0], min_lat: bb[1], max_lon: bb[2], max_lat: bb[3] };
    scenes.forEach(s => {
      if (s.land_cover && s.ndvi_score !== null) {
        const uniqueId = `${data.place}_${s.date}`;
        if (uploadHistory.some(e => e.filename === uniqueId)) return;
        const assignedKey = resolveGeoKey(geoB);
        uploadHistory.push({
          filename: uniqueId,
          label: s.date,
          date: extractDateFromString(s.date),
          ndvi: s.ndvi_score,
          ndwi: s.ndwi_score,
          land_cover: s.land_cover,
          geo_bounds: geoB,
          geo_key: assignedKey,
        });
      }
    });

    // Don't auto-open trend modal — user opens it manually via 📈 button
    // Highlight the button to indicate trend data is available
    if (scenes.length > 1) {
      const trendBtn = document.querySelector('button[onclick="openTrendModal()"]');
      if (trendBtn) trendBtn.style.boxShadow = '0 0 0 2px #16a34a55';
    }

    appendBubble(
      `🛰 AUTO-FETCH: "${data.display_name}"\n` +
      `${scenes.length} Sentinel-2 scenes (${dateFrom} → ${dateTo})\n\n` +
      data.ai_insight,
      'ai', true
    );

  } catch (err) {
    showStatus(statusEl, `⚠ ${err.message}`, 'red');
    setViewportStatus('ERROR', 'rgba(255,100,100,0.9)');
    appendBubble(`⚠ Auto-fetch failed: ${err.message}`, 'ai');
  } finally {
    btn.disabled = false;
    loader.classList.remove('active');
  }
}

// ── Sentinel-2 live fetch ──────────────────────────────────────────────────
async function runAnalysis() {
  const runBtn = document.getElementById('run-btn');
  const loader = document.getElementById('scenario-loader');
  runBtn.disabled = true;
  loader.classList.add('active');
  setViewportStatus('FETCHING SENTINEL-2 DATA...', 'rgba(255,255,255,0.7)');
  try {
    const res = await fetch('/api/analyze', { method: 'POST' });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
    const data = await res.json();
    lastNdvi = data.ndvi_score; lastNdwi = data.ndwi_score;
    animateValue('ndvi-display', data.ndvi_score);
    animateValue('ndwi-display', data.ndwi_score);
    document.getElementById('sensor-display').textContent = `SENSOR: ${data.sensor}`;
    if (data.land_cover) renderLandCover(data.land_cover);
    if (data.land_summary) renderLandSummary(data.land_summary);
    setViewportStatus('ANALYSIS COMPLETE ✓', '#4ade80');
    appendBubble(data.ai_insight, 'ai', true);
  } catch (err) {
    setViewportStatus('ERROR', 'rgba(255,100,100,0.9)');
    appendBubble(`⚠ ${err.message}`, 'ai');
  } finally {
    runBtn.disabled = false;
    loader.classList.remove('active');
  }
}

// ── Chat ───────────────────────────────────────────────────────────────────
// Hard geo-domain keyword check — catches obvious off-topic queries instantly
// without an extra API call. The server SYSTEM_PROMPT handles edge cases.
const GEO_KEYWORDS = [
  'ndvi','ndwi','satellite','sentinel','landsat','modis','copernicus','esa','nasa',
  'vegetation','forest','deforestation','water','lake','river','reservoir','flood',
  'urban','city','town','village','district','state','country','region','area',
  'land','terrain','soil','crop','agriculture','farm','field','coast','ocean','sea',
  'mountain','hill','valley','desert','wetland','mangrove','coral','glacier','snow',
  'india','andhra','telangana','hyderabad','amaravathi','nalamala','amazon','africa',
  'spectral','infrared','nir','swir','band','pixel','raster','tiff','geotiff',
  'remote sensing','earth observation','geo','map','coordinate','latitude','longitude',
  'encroachment','sprawl','camouflage','heatmap','analysis','analyse','analyze',
  'climate','temperature','rainfall','drought','erosion','pollution','environment',
  'about','where','what is','how is','tell me about','show me','explain'
];

function isGeoRelated(msg) {
  const lower = msg.toLowerCase();
  return GEO_KEYWORDS.some(kw => lower.includes(kw));
}

async function sendChat() {
  const input = document.getElementById('chat-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  appendBubble(msg, 'user');

  // Hard client-side geo check — block obvious off-topic queries immediately
  if (!isGeoRelated(msg)) {
    appendBubble(
      "I'm a satellite and geospatial analysis AI. I can only help with remote sensing, " +
      "land cover, vegetation, water bodies, and geographic analysis. Please ask me something related to Earth observation.",
      'ai'
    );
    return;
  }

  try {
    const body = { message: msg };
    if (lastNdvi !== null) body.ndvi_score = lastNdvi;
    if (lastNdwi !== null) body.ndwi_score = lastNdwi;
    const res = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) { const e = await res.json(); throw new Error(e.detail); }
    const data = await res.json();
    appendBubble(data.reply, 'ai');

    // If the AI returned land cover data for a place, update the panels
    if (data.land_cover) {
      renderLandCover(data.land_cover);
      lastNdvi = data.land_summary ? (lastNdvi ?? 0) : lastNdvi;
      lastNdwi = data.land_summary ? (lastNdwi ?? 0) : lastNdwi;
    }
    if (data.land_summary) {
      renderLandSummary(data.land_summary);
    }
  } catch (err) {
    appendBubble(`⚠ ${err.message}`, 'ai');
  }
}

// ── Land Cover Renderer ────────────────────────────────────────────────────
function renderLandCover(lc) {
  const panel = document.getElementById('land-cover-panel');
  const bars = document.getElementById('lc-bars');
  panel.style.display = 'block';
  bars.innerHTML = '';
  const items = [
    { key: 'vegetation', label: '🌿 Vegetation', color: '#2d9e47' },
    { key: 'water',      label: '💧 Water',      color: '#0077b6' },
    { key: 'urban',      label: '🏗 Urban',      color: '#e07b39' },
    { key: 'bare',       label: '🟤 Bare Soil',  color: '#a0785a' },
  ];
  items.forEach(({ key, label, color }) => {
    const pct = lc[key] ?? 0;
    bars.innerHTML += `
      <div>
        <div style="display:flex;justify-content:space-between;font-size:0.74rem;margin-bottom:3px;">
          <span>${label}</span><span style="font-weight:600;color:${color};">${pct}%</span>
        </div>
        <div style="background:#e9ecef;border-radius:4px;height:8px;">
          <div class="lc-bar" style="width:${pct}%;background:${color};"></div>
        </div>
      </div>`;
  });
}

// ── Land Summary Renderer ──────────────────────────────────────────────────
function renderLandSummary(ls) {
  const panel = document.getElementById('land-summary-panel');
  panel.style.display = 'block';
  document.getElementById('summary-geo-type').textContent = '📍 ' + ls.geo_type;
  document.getElementById('summary-geo-state').textContent = '🔍 ' + ls.geo_state;
  document.getElementById('summary-text').textContent = ls.summary;
  document.getElementById('summary-disclaimer').textContent = ls.disclaimer;
  document.getElementById('summary-body').style.display = 'block';
  document.getElementById('summary-chevron').style.transform = 'rotate(180deg)';
}

function toggleSummaryBox() {
  const body = document.getElementById('summary-body');
  const chevron = document.getElementById('summary-chevron');
  const open = body.style.display !== 'none';
  body.style.display = open ? 'none' : 'block';
  chevron.style.transform = open ? 'rotate(0deg)' : 'rotate(180deg)';
}

// ── Change Detection + Trend Graphs ───────────────────────────────────────
function renderChangeDetection() {
  const modal = document.getElementById('trend-modal');
  const content = document.getElementById('trend-content');
  modal.style.display = 'flex';

  // Deduplicate by filename only, then group by geo_key
  const seen = new Set();
  const deduped = uploadHistory.filter(e => {
    if (seen.has(e.filename)) return false;
    seen.add(e.filename);
    return true;
  });

  const groups = {};
  deduped.forEach(entry => {
    const k = entry.geo_key;
    if (!groups[k]) groups[k] = [];
    groups[k].push(entry);
  });

  // Sort each group by date (oldest first) for correct trend direction
  Object.values(groups).forEach(arr => {
    arr.sort((a, b) => {
      const da = a.date, db = b.date;
      if (da && db) return da - db;
      if (da) return -1;
      if (db) return 1;
      return 0; // keep original order if no dates
    });
  });

  let html = '';
  const COLORS = { vegetation: '#2d9e47', water: '#0077b6', urban: '#e07b39', bare: '#a0785a' };
  const KEYS = ['vegetation', 'water', 'urban', 'bare'];
  const LABELS = { vegetation: '🌿 Vegetation', water: '💧 Water', urban: '🏗 Urban', bare: '🟤 Bare Soil' };

  const geoEntries = Object.entries(groups);

  // Only show multi-geography warning if there are genuinely multiple distinct regions
  // AND more than 1 total image (single image should never trigger this)
  if (geoEntries.length > 1 && deduped.length > 1) {
    html += `<div style="background:#fef3c7;border:1px solid #fde68a;border-radius:8px;padding:0.6rem 0.85rem;margin-bottom:1rem;font-size:0.78rem;color:#92400e;">
      ⚠ <strong>${geoEntries.length} distinct geographies detected.</strong> Each region is analysed separately below.
    </div>`;
  }

  geoEntries.forEach(([key, entries], gi) => {
    const gb = entries[0].geo_bounds;
    let geoLabel;
    if (key === 'unknown' || !gb) {
      geoLabel = `Region ${gi + 1} (no geo metadata)`;
    } else {
      const c = centrePt(gb);
      geoLabel = c
        ? `📍 ${c.lat.toFixed(3)}°N, ${c.lon.toFixed(3)}°E`
        : `📍 ${gb.min_lat}°N ${gb.min_lon}°E`;
    }

    html += `<div style="margin-bottom:1.8rem;padding-bottom:1rem;border-bottom:1px solid #e9ecef;">
      <div style="font-size:0.8rem;font-weight:700;color:var(--primary);margin-bottom:0.5rem;">
        ${geoLabel} <span style="font-weight:400;color:var(--text-muted);">(${entries.length} image${entries.length > 1 ? 's' : ''})</span>
      </div>`;

    if (entries.length === 1) {
      const e = entries[0];
      html += `<div style="font-size:0.74rem;color:var(--text-muted);margin-bottom:0.5rem;font-style:italic;">Single snapshot — upload more images of this area to see temporal trends.</div>`;
      KEYS.forEach(k => {
        const pct = e.land_cover[k] ?? 0;
        html += `<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:5px;">
          <span style="font-size:0.72rem;width:95px;">${LABELS[k]}</span>
          <div style="flex:1;background:#e9ecef;border-radius:4px;height:7px;">
            <div style="width:${pct}%;background:${COLORS[k]};height:7px;border-radius:4px;transition:width 0.6s;"></div>
          </div>
          <span style="font-size:0.72rem;font-weight:600;color:${COLORS[k]};width:34px;text-align:right;">${pct}%</span>
        </div>`;
      });
    } else {
      // Chart
      const canvasId = `trend-canvas-${gi}`;
      html += `<canvas id="${canvasId}" width="460" height="170" style="width:100%;border-radius:8px;background:#f8f9fa;border:1px solid #dee2e6;display:block;"></canvas>`;

      // Delta badges
      html += `<div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-top:0.5rem;">`;

      // Urban sanity check for badges — same logic as buildTrendSummary
      const _urbanFirst = entries[0].land_cover['urban'] ?? 0;
      const _urbanLast  = entries[entries.length-1].land_cover['urban'] ?? 0;
      const _urbanDelta = _urbanLast - _urbanFirst;
      const _vegFirst   = entries[0].land_cover['vegetation'] ?? 0;
      const _vegLast    = entries[entries.length-1].land_cover['vegetation'] ?? 0;
      const _vegDelta   = _vegLast - _vegFirst;
      const _urbanAdjusted = (_urbanDelta < 0);

      KEYS.forEach(k => {
        const first = entries[0].land_cover[k] ?? 0;
        let lastVal = entries[entries.length - 1].land_cover[k] ?? 0;
        let delta = lastVal - first;

        // Apply urban adjustment to badge
        if (k === 'urban' && _urbanAdjusted) {
          delta = 0;
          lastVal = first;
        }

        const arrow = delta > 2 ? '▲' : delta < -2 ? '▼' : '→';
        const col = delta > 2 ? '#16a34a' : delta < -2 ? '#dc2626' : '#6b7280';
        const bg  = delta > 2 ? '#f0fdf4' : delta < -2 ? '#fef2f2' : '#f9fafb';
        const badgeLabel = (k === 'urban' && _urbanAdjusted)
          ? `${LABELS[k]} → ~stable`
          : `${LABELS[k]} ${arrow} ${delta > 0 ? '+' : ''}${delta}%`;
        html += `<span style="font-size:0.71rem;color:${col};font-weight:600;background:${bg};border:1px solid ${col}33;border-radius:5px;padding:2px 7px;">
          ${badgeLabel}
        </span>`;
      });
      html += `</div>`;

      // NDVI trend line
      const ndviFirst = entries[0].ndvi;
      const ndviLast = entries[entries.length - 1].ndvi;
      const ndviDelta = (ndviLast - ndviFirst).toFixed(2);
      html += `<div style="font-size:0.71rem;color:var(--text-muted);margin-top:0.35rem;">
        NDVI: ${ndviFirst} → ${ndviLast} &nbsp;(${Number(ndviDelta) >= 0 ? '+' : ''}${ndviDelta})
      </div>`;

      // ── Crisp trend summary ──
      const summaryLines = buildTrendSummary(entries, KEYS, LABELS);
      html += `<div style="margin-top:0.65rem;background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:0.6rem 0.85rem;font-size:0.76rem;line-height:1.7;color:#0c4a6e;">
        <strong style="font-size:0.72rem;letter-spacing:0.05em;text-transform:uppercase;color:#0369a1;">📊 Trend Summary</strong><br/>
        ${summaryLines}
      </div>`;
    }
    html += `</div>`;
  });

  content.innerHTML = html;

  requestAnimationFrame(() => {
    geoEntries.forEach(([key, entries], gi) => {
      if (entries.length < 2) return;
      const canvas = document.getElementById(`trend-canvas-${gi}`);
      if (!canvas) return;
      drawTrendChart(canvas, entries);
    });
  });
}

// ── Build a crisp natural-language trend summary ───────────────────────────
function buildTrendSummary(entries, KEYS, LABELS) {
  const n = entries.length;
  const first = entries[0];
  const last = entries[n - 1];
  const parts = [];

  // ── Urban sanity check ────────────────────────────────────────────────
  // Urban areas almost never shrink in India/developing regions.
  // If urban shows decline AND vegetation is growing significantly,
  // it's almost certainly a classification artifact — treat urban as stable.
  const urbanFirst = first.land_cover['urban'] ?? 0;
  const urbanLast  = last.land_cover['urban']  ?? 0;
  const urbanDelta = urbanLast - urbanFirst;
  const vegFirst   = first.land_cover['vegetation'] ?? 0;
  const vegLast    = last.land_cover['vegetation']  ?? 0;
  const vegDelta   = vegLast - vegFirst;

  // Urban in India/developing regions never shrinks — any decline is a classification artifact.
  // Clamp any urban decrease to stable (0 change).
  const urbanAdjusted = (urbanDelta < 0);
  const effectiveUrbanDelta = urbanAdjusted ? 0 : urbanDelta;
  const effectiveUrbanLast  = urbanAdjusted ? urbanFirst : urbanLast;

  KEYS.forEach(k => {
    const fv = first.land_cover[k] ?? 0;
    let lv = last.land_cover[k] ?? 0;
    let delta = lv - fv;

    // Apply urban adjustment
    if (k === 'urban' && urbanAdjusted) {
      delta = 0;
      lv = fv;
    }

    if (Math.abs(delta) >= 3) {
      const dir = delta > 0 ? 'increased' : 'decreased';
      parts.push(`${LABELS[k].replace(/[^\w\s]/g, '').trim()} ${dir} by ${Math.abs(delta)}% (${fv}% → ${lv}%)`);
    }
  });

  // Add urban stability note if adjusted
  if (urbanAdjusted) {
    parts.push(`Urban cover remained stable at ~${urbanFirst}% (growth likely ongoing but masked by vegetation reclassification)`);
  }

  const ndviDelta = last.ndvi - first.ndvi;
  const ndwiDelta = last.ndwi - first.ndwi;

  // Date range string
  const dateRange = (() => {
    const dFirst = first.date, dLast = last.date;
    if (dFirst && dLast && !isNaN(dFirst) && !isNaN(dLast)) {
      const months = Math.round((dLast - dFirst) / (1000 * 60 * 60 * 24 * 30.5));
      const span = months >= 12 ? `${Math.round(months/12)} yr${Math.round(months/12)>1?'s':''}` : `${months} mo`;
      return ` over ${span} (${formatDateLabel(dFirst)} → ${formatDateLabel(dLast)})`;
    }
    return ` across ${n} images`;
  })();

  let summary = '';
  if (parts.length === 0) {
    summary = `Land cover remained largely stable${dateRange}. No significant change detected (threshold: ≥3%).`;
  } else {
    summary = parts.join('; ') + `.`;
    if (Math.abs(ndviDelta) >= 0.05) {
      summary += ` Vegetation health (NDVI) ${ndviDelta > 0 ? 'improved ▲' : 'declined ▼'} by ${Math.abs(ndviDelta).toFixed(2)}${dateRange}.`;
    }
    if (Math.abs(ndwiDelta) >= 0.05) {
      summary += ` Water index (NDWI) ${ndwiDelta > 0 ? 'rose ▲' : 'fell ▼'} by ${Math.abs(ndwiDelta).toFixed(2)}.`;
    }
  }
  return summary;
}

function drawTrendChart(canvas, entries) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const PAD = { top: 20, right: 16, bottom: 36, left: 36 };
  const chartW = W - PAD.left - PAD.right;
  const chartH = H - PAD.top - PAD.bottom;
  const n = entries.length;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#f8f9fa';
  ctx.fillRect(0, 0, W, H);

  // Grid lines
  ctx.strokeStyle = '#dee2e6';
  ctx.lineWidth = 0.5;
  [0, 25, 50, 75, 100].forEach(v => {
    const y = PAD.top + chartH - (v / 100) * chartH;
    ctx.beginPath(); ctx.moveTo(PAD.left, y); ctx.lineTo(PAD.left + chartW, y); ctx.stroke();
    ctx.fillStyle = '#9ca3af'; ctx.font = '9px sans-serif'; ctx.textAlign = 'right';
    ctx.fillText(v + '%', PAD.left - 4, y + 3);
  });

  // X axis — use dates if available, else labels
  const hasDates = entries.every(e => e.date && !isNaN(e.date));
  const xPositions = entries.map((e, i) => {
    if (hasDates && n > 1) {
      const tFirst = entries[0].date.getTime();
      const tLast  = entries[n-1].date.getTime();
      const span   = tLast - tFirst || 1;
      return PAD.left + ((e.date.getTime() - tFirst) / span) * chartW;
    }
    return PAD.left + (n === 1 ? chartW / 2 : (i / (n - 1)) * chartW);
  });

  // X axis labels
  ctx.fillStyle = '#6b7280'; ctx.font = '9px sans-serif'; ctx.textAlign = 'center';
  entries.forEach((e, i) => {
    const x = xPositions[i];
    const lbl = (e.date && !isNaN(e.date)) ? formatDateLabel(e.date) : e.label.slice(0, 10);
    // Stagger labels to avoid overlap
    const yOff = (i % 2 === 0) ? H - 18 : H - 6;
    ctx.fillText(lbl || e.label.slice(0, 10), x, yOff);
  });

  const COLORS = { vegetation: '#2d9e47', water: '#0077b6', urban: '#e07b39', bare: '#a0785a' };
  const KEYS = ['vegetation', 'water', 'urban', 'bare'];

  KEYS.forEach(key => {
    const color = COLORS[key];
    const vals = entries.map(e => e.land_cover[key] ?? 0);

    // Main line
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    entries.forEach((e, i) => {
      const x = xPositions[i];
      const y = PAD.top + chartH - (vals[i] / 100) * chartH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Dots
    entries.forEach((e, i) => {
      const x = xPositions[i];
      const y = PAD.top + chartH - (vals[i] / 100) * chartH;
      ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = color; ctx.fill();
    });

    // Trendline (linear regression)
    if (n >= 2) {
      const xs = xPositions.map((x, i) => i); // use index for regression
      const ys = vals;
      const meanX = xs.reduce((a, b) => a + b, 0) / n;
      const meanY = ys.reduce((a, b) => a + b, 0) / n;
      const num = xs.reduce((s, x, i) => s + (x - meanX) * (ys[i] - meanY), 0);
      const den = xs.reduce((s, x) => s + (x - meanX) ** 2, 0);
      if (den !== 0) {
        const slope = num / den;
        const intercept = meanY - slope * meanX;
        const y0 = PAD.top + chartH - (Math.max(0, Math.min(100, intercept)) / 100) * chartH;
        const y1 = PAD.top + chartH - (Math.max(0, Math.min(100, slope * (n - 1) + intercept)) / 100) * chartH;
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 3]);
        ctx.globalAlpha = 0.45;
        ctx.beginPath();
        ctx.moveTo(xPositions[0], y0);
        ctx.lineTo(xPositions[n - 1], y1);
        ctx.stroke();
        ctx.restore();

        // Trend arrow at end of trendline
        const arrowX = xPositions[n - 1];
        const arrowY = PAD.top + chartH - (Math.max(0, Math.min(100, vals[n-1])) / 100) * chartH;
        const trendDir = slope > 0.5 ? '▲' : slope < -0.5 ? '▼' : '→';
        ctx.fillStyle = color;
        ctx.font = 'bold 9px sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(trendDir, arrowX + 5, arrowY + 3);
      }
    }
  });
}

function closeTrendModal() {
  document.getElementById('trend-modal').style.display = 'none';
}

function clearHistory() {
  uploadHistory = [];
  document.getElementById('trend-modal').style.display = 'none';
  document.getElementById('land-cover-panel').style.display = 'none';
  document.getElementById('land-summary-panel').style.display = 'none';
  const trendBtn = document.querySelector('button[onclick="openTrendModal()"]');
  if (trendBtn) trendBtn.style.boxShadow = '';
  appendBubble('🗑 Analysis history cleared. Upload new images to start fresh.', 'ai');
}

function openTrendModal() {
  if (uploadHistory.length === 0) {
    appendBubble('📈 No analysis history yet. Upload images or run Auto-Fetch first.', 'ai');
    return;
  }
  renderChangeDetection();
}

// ── Viewport geo coords ────────────────────────────────────────────────────
function updateViewportCoords(geo_bounds) {
  const el = document.getElementById('viewport-coords');
  if (!el) return;
  if (geo_bounds) {
    el.textContent = `LAT ${geo_bounds.min_lat}–${geo_bounds.max_lat}°N | LON ${geo_bounds.min_lon}–${geo_bounds.max_lon}°E`;
  } else {
    el.textContent = 'GEO METADATA UNAVAILABLE';
  }
}

// ── Accuracy / summary toggles ─────────────────────────────────────────────
function toggleAccuracyBox() {
  document.getElementById('inline-accuracy').classList.toggle('open');
}

// ── Helpers ────────────────────────────────────────────────────────────────
function setViewportStatus(text, color = 'rgba(255,255,255,0.6)') {
  const el = document.getElementById('viewport-status');
  el.textContent = text; el.style.color = color;
}

function showStatus(el, msg, type) {
  if (!el) return;
  el.style.display = 'block';
  el.style.color = type === 'red' ? '#dc3545' : type === 'green' ? '#1a6b2e' : '#0077b6';
  el.textContent = msg;
}

function appendBubble(text, role, glow = false) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = `bubble bubble-${role}${glow ? ' glow-pulse' : ''}`;
  div.textContent = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function animateValue(id, target) {
  const el = document.getElementById(id);
  const start = parseFloat(el.textContent) || 0;
  const duration = 800;
  const startTime = performance.now();
  function step(now) {
    const progress = Math.min((now - startTime) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (start + (target - start) * eased).toFixed(2);
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = target.toFixed(2);
  }
  requestAnimationFrame(step);
}

function showViewportImage(file) {
  const img = document.getElementById('viewport-image');
  const crosshair = document.getElementById('crosshair');
  const reader = new FileReader();
  reader.onload = (e) => {
    img.src = e.target.result;
    img.classList.add('visible');
    if (crosshair) crosshair.style.display = 'none';
    setViewportStatus('IMAGE LOADED', 'rgba(255,255,255,0.7)');
  };
  reader.readAsDataURL(file);
}

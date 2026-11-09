"""HTTP inference service.

    NEURODX_CHECKPOINT=runs/alzheimer/<run>/best.pt uvicorn neurodx.serve:app --port 8000

Endpoints
  GET  /          minimal upload page (demo)
  GET  /health    liveness + model metadata
  POST /predict   multipart field "file"; query params explain=bool, abstain_below=float
"""
from __future__ import annotations

import base64
import io
import os
import threading

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from .inference import LoadedModel, load_checkpoint, overlay, predict

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 4096 * 4096
DISCLAIMER = "Research prototype. Not a medical device; not for clinical use."


def create_app(checkpoint: str | None = None, default_abstain: float | None = None) -> FastAPI:
    checkpoint = checkpoint or os.environ.get("NEURODX_CHECKPOINT")
    default_abstain = default_abstain if default_abstain is not None else float(
        os.environ.get("NEURODX_ABSTAIN_BELOW", "0.6"))
    app = FastAPI(title="neurodx", version="0.2.0", description=DISCLAIMER)
    state: dict[str, LoadedModel | None] = {"model": None}
    lock = threading.Lock()  # one forward pass at a time (Grad-CAM mutates grads)

    def model() -> LoadedModel:
        if state["model"] is None:
            if not checkpoint:
                raise HTTPException(503, "No model configured; set NEURODX_CHECKPOINT.")
            state["model"] = load_checkpoint(checkpoint)
        return state["model"]

    @app.get("/health")
    def health() -> dict:
        try:
            lm = model()
        except HTTPException as e:
            return {"status": "no_model", "detail": e.detail}
        return {"status": "ok", "classes": lm.classes, "temperature": lm.temperature,
                "device": str(lm.device), **lm.meta}

    @app.post("/predict")
    async def predict_endpoint(
        file: UploadFile = File(...),
        explain: bool = Query(False),
        abstain_below: float | None = Query(None, ge=0.0, le=1.0),
    ) -> dict:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"File larger than {MAX_UPLOAD_BYTES // 2**20} MB.")
        try:
            image = Image.open(io.BytesIO(data))
            if image.width * image.height > MAX_PIXELS:
                raise HTTPException(413, "Image dimensions too large.")
            image.load()
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            raise HTTPException(415, "Unsupported or corrupt image (expected PNG/JPEG).") from e
        lm = model()
        with lock:
            result = predict(lm, image, default_abstain if abstain_below is None else abstain_below, explain)
        if "cam" in result:
            buf = io.BytesIO()
            overlay(image, result.pop("cam")).save(buf, format="PNG")
            result["gradcam_png_base64"] = base64.b64encode(buf.getvalue()).decode()
        result["disclaimer"] = DISCLAIMER
        return result

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _INDEX_HTML

    return app


_INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>neurodx demo</title>
<style>body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;padding:0 16px}
.warn{background:#fff4e5;border:1px solid #f0b46a;padding:.6rem;border-radius:6px}
img{max-width:100%;border-radius:6px}td{padding:2px 10px}</style></head><body>
<h1>neurodx: MRI slice classifier</h1>
<p class="warn">Research prototype. Not a medical device; not for clinical use.</p>
<form id="f"><input type="file" name="file" accept="image/png,image/jpeg" required>
<label><input type="checkbox" id="x" checked> Grad-CAM</label> <button>Predict</button></form>
<div id="out"></div>
<script>
document.getElementById('f').onsubmit = async (e) => {
  e.preventDefault();
  const out = document.getElementById('out'); out.textContent = 'Running...';
  const r = await fetch('/predict?explain=' + document.getElementById('x').checked,
                        {method: 'POST', body: new FormData(e.target)});
  const j = await r.json();
  if (!r.ok) { out.textContent = 'Error: ' + (j.detail || r.status); return; }
  const rows = Object.entries(j.probabilities).map(([k, v]) =>
    '<tr><td>' + k + '</td><td>' + (100 * v).toFixed(1) + '%</td></tr>').join('');
  out.innerHTML = '<h2>' + j.prediction + ' (' + (100 * j.confidence).toFixed(1) + '%)' +
    (j.abstained ? ' - low confidence, refer for review' : '') + '</h2><table>' + rows + '</table>' +
    (j.gradcam_png_base64 ? '<img alt="Grad-CAM" src="data:image/png;base64,' + j.gradcam_png_base64 + '">' : '');
};
</script></body></html>"""

app = create_app()

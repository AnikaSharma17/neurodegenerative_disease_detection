import base64
import io
import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from neurodx import predict as predict_cli
from neurodx import serve
from neurodx.inference import grad_cam, load_checkpoint, predict, preprocess
from neurodx.models import build_model


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    model = build_model("multitask", "mobilenet_v2", 2, pretrained=False)
    path = tmp_path_factory.mktemp("ckpt") / "best.pt"
    torch.save({"model_state": model.state_dict(), "classes": ["normal", "parkinson"], "epoch": 1,
                "temperature": 1.5,
                "args": {"arch": "multitask", "backbone": "mobilenet_v2", "image_size": 64,
                         "disease": "parkinson"}}, path)
    return path


def _png(size=(80, 96)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, size[::-1], dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def test_predict_returns_calibrated_distribution(checkpoint):
    lm = load_checkpoint(checkpoint, "cpu")
    r = predict(lm, Image.open(io.BytesIO(_png())), abstain_below=0.99, explain=True)
    assert set(r["probabilities"]) == {"normal", "parkinson"}
    assert abs(sum(r["probabilities"].values()) - 1) < 1e-5
    assert r["abstained"] is True
    assert r["cam"].shape == (64, 64)
    assert 0 <= r["cam"].min() and r["cam"].max() <= 1


def test_grad_cam_is_deterministic(checkpoint):
    lm = load_checkpoint(checkpoint, "cpu")
    x = preprocess(Image.open(io.BytesIO(_png())), 64)
    a, _ = grad_cam(lm.model, x)
    b, _ = grad_cam(lm.model, x)
    assert np.allclose(a, b)


def test_predict_cli_writes_cams(checkpoint, tmp_path):
    img = tmp_path / "slice.png"
    img.write_bytes(_png())
    out = predict_cli.main(["--checkpoint", str(checkpoint), "--device", "cpu", "--cam-dir", str(tmp_path / "cams"),
                            str(img)])
    assert out[0]["prediction"] in {"normal", "parkinson"}
    assert (tmp_path / "cams" / "cam_slice.png").is_file()


def test_api(checkpoint, monkeypatch):
    client = TestClient(serve.create_app(str(checkpoint), default_abstain=0.0))
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["classes"] == ["normal", "parkinson"]
    assert "not for clinical use" in client.get("/").text

    r = client.post("/predict?explain=true", files={"file": ("x.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {"normal", "parkinson"} and body["abstained"] is False
    Image.open(io.BytesIO(base64.b64decode(body["gradcam_png_base64"])))  # valid PNG
    json.dumps(body)  # JSON-serialisable

    assert client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")}).status_code == 415
    monkeypatch.setattr(serve, "MAX_UPLOAD_BYTES", 100)
    assert client.post("/predict", files={"file": ("x.png", _png(), "image/png")}).status_code == 413


def test_api_without_model_reports_unavailable():
    client = TestClient(serve.create_app(checkpoint=None))
    serve_env = client.get("/health").json()
    assert serve_env["status"] == "no_model"
    assert client.post("/predict", files={"file": ("x.png", _png(), "image/png")}).status_code == 503

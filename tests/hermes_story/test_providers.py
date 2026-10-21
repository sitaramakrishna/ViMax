import asyncio
import base64
import io

from PIL import Image

import tools.image_generator_openrouter_api as m
from hermes_story import providers


def _png(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_aspect_ratio_sent_in_body(monkeypatch):
    seen = {}

    async def fake_post(url, *, headers, payload, timeout):
        seen.update(payload)
        return 200, {"data": [{"b64_json": _png(32, 18)}]}

    monkeypatch.setattr(providers, "_original_post_json", fake_post)
    gen = providers.OpenRouterImage(api_key="k", model="google/gemini-2.5-flash-image")
    asyncio.run(gen.generate_single_image("a robot"))
    assert seen["aspect_ratio"] == "16:9"
    assert seen["model"] == "google/gemini-2.5-flash-image"


def test_plain_driver_unaffected(monkeypatch):
    seen = {}

    async def fake_post(url, *, headers, payload, timeout):
        seen.update(payload)
        return 200, {"data": [{"b64_json": _png(32, 18)}]}

    monkeypatch.setattr(providers, "_original_post_json", fake_post)
    gen = m.ImageGeneratorOpenRouterAPI(api_key="k", model="x")
    asyncio.run(gen.generate_single_image("a robot"))
    assert "aspect_ratio" not in seen

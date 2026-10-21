"""Provider adapters that fix ViMax driver gaps without editing ViMax files.

ViMax's OpenRouter image driver asks for a landscape picture only in the prompt
text. Models such as ``google/gemini-2.5-flash-image`` ignore that and return
squares, which ViMax then rejects for first frames. OpenRouter's ``/images``
endpoint accepts an ``aspect_ratio`` field, so this adapter adds it to the
request body.
"""

from __future__ import annotations

import contextvars

import tools.image_generator_openrouter_api as _openrouter_image

#: Extra request-body fields for the OpenRouter ``/images`` call in progress.
_EXTRA_FIELDS: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "hermes_story_openrouter_image_extra", default={}
)

_original_post_json = _openrouter_image._post_json


async def _post_json_with_extra(url, *, payload, **kwargs):
    extra = _EXTRA_FIELDS.get()
    if extra:
        payload = {**payload, **extra}
    return await _original_post_json(url, payload=payload, **kwargs)


# Installed once; it only adds fields while an adapter call has set them.
if getattr(_openrouter_image._post_json, "__name__", "") != "_post_json_with_extra":
    _openrouter_image._post_json = _post_json_with_extra


class OpenRouterImage(_openrouter_image.ImageGeneratorOpenRouterAPI):
    """ViMax's OpenRouter image driver, sending ``aspect_ratio`` to the API."""

    def __init__(self, *args, default_aspect_ratio: str = "16:9", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.default_aspect_ratio = default_aspect_ratio

    async def generate_single_image(self, prompt, reference_image_paths=None, aspect_ratio=None, **kwargs):
        ratio = aspect_ratio or self.default_aspect_ratio
        token = _EXTRA_FIELDS.set({"aspect_ratio": ratio})
        try:
            return await super().generate_single_image(
                prompt, reference_image_paths, aspect_ratio=ratio, **kwargs
            )
        finally:
            _EXTRA_FIELDS.reset(token)

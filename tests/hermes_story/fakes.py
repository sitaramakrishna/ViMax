"""Fake providers for hermes_story tests.  None of them touch the network."""

from __future__ import annotations

from PIL import Image
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from interfaces.image_output import ImageOutput
from interfaces.video_output import VideoOutput


class FakeImageGen:
    """Returns a tiny PIL image and records every call."""

    def __init__(self, raise_exc: BaseException | None = None) -> None:
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "reference_image_paths": list(reference_image_paths or []),
                **kwargs,
            }
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (4, 4), (200, 30, 30)))


class FakeVideoGen:
    """Returns an ``mp4`` byte output and records every call."""

    def __init__(self, raise_exc: BaseException | None = None) -> None:
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def generate_single_video(self, prompt="", reference_image_paths=None, **kwargs):
        self.calls.append(
            {
                "prompt": prompt,
                "reference_image_paths": list(reference_image_paths or []),
                **kwargs,
            }
        )
        if self.raise_exc is not None:
            raise self.raise_exc
        return VideoOutput(fmt="bytes", ext="mp4", data=b"mp4")


class FakeChat(FakeListChatModel):
    """A queued-response chat model usable with ``chat_model | parser``."""

    def __init__(self, responses: list[str] | None = None, **kwargs) -> None:
        super().__init__(responses=list(responses or ["ok"]), **kwargs)

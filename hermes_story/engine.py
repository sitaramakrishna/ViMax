"""Builds the chat model, guarded generators and ViMax pipelines for one film.

Every project gets its own :class:`~hermes_story.ledger.Ledger`, so the paid
generators handed to the ViMax pipelines always charge the right film and stop
at that film's cap.  The real providers are only built when the caller does not
inject its own (tests inject fakes and never touch the network).
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain.chat_models import init_chat_model

from hermes_story import guarded, project
from hermes_story.config import StoryConfig
from hermes_story.ledger import Ledger
from pipelines.idea2video_pipeline import Idea2VideoPipeline
from pipelines.script2video_pipeline import Script2VideoPipeline
from hermes_story.providers import OpenRouterImage
from tools.video_generator_openrouter_api import VideoGeneratorOpenRouterAPI


def _cap(pid: str, cfg: StoryConfig) -> float:
    """The film's cap from ``ledger.json``, else the configured default."""
    data = project.read_json(project.path(pid) / "ledger.json", None)
    if isinstance(data, dict):
        try:
            cap = float(data.get("cap_usd"))
        except (TypeError, ValueError):
            cap = None
        if cap is not None:
            return cap
    return float(cfg.default_cap_usd)


class Engine:
    """Builds the ViMax pipelines for ``pid``, all charging ``self.ledger``."""

    def __init__(
        self,
        pid: str,
        cfg: StoryConfig,
        *,
        chat=None,
        image=None,
        sheet_image=None,
        video=None,
    ) -> None:
        self.pid = pid
        self.cfg = cfg
        self.project_dir: Path = project.path(pid)
        self.ledger = Ledger(pid, _cap(pid, cfg))

        self.chat = chat if chat is not None else self._build_chat()
        self.image = guarded.GuardedImageGenerator(
            image if image is not None else self._build_image(),
            self.ledger,
            price=cfg.price_image_call,
            model=cfg.image_model,
        )
        # Character sheets use their own model.  An injected ``image`` stands in
        # for it too when no separate ``sheet_image`` is given (tests).
        if sheet_image is None:
            sheet_image = image if image is not None else self._build_sheet_image()
        self.sheet_image = guarded.GuardedImageGenerator(
            sheet_image,
            self.ledger,
            price=cfg.price_sheet_image_call,
            model=cfg.sheet_image_model,
        )
        self.video = guarded.GuardedVideoGenerator(
            video if video is not None else self._build_video(),
            self.ledger,
            price_per_s=cfg.price_video_second,
            duration_s=cfg.video_duration_s,
            model=cfg.video_model,
        )

    # -- providers -------------------------------------------------------

    def _build_chat(self):
        return init_chat_model(
            model=self.cfg.chat_model,
            model_provider="openai",
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key(),
        )

    def _build_image(self):
        return OpenRouterImage(
            api_key=self.cfg.api_key(), model=self.cfg.image_model
        )

    def _build_sheet_image(self):
        return OpenRouterImage(
            api_key=self.cfg.api_key(),
            model=self.cfg.sheet_image_model,
            quality=self.cfg.sheet_image_quality,
        )

    def _build_video(self):
        # The video generator reads its request shape from the environment.
        os.environ["VIMAX_OPENROUTER_VIDEO_DURATION"] = str(self.cfg.video_duration_s)
        os.environ["VIMAX_OPENROUTER_VIDEO_RESOLUTION"] = self.cfg.video_resolution
        os.environ["VIMAX_OPENROUTER_GENERATE_AUDIO"] = "0"
        return VideoGeneratorOpenRouterAPI(
            api_key=self.cfg.api_key(), model=self.cfg.video_model
        )

    # -- pipelines -------------------------------------------------------

    def idea(self) -> Idea2VideoPipeline:
        """The whole-film pipeline, working in ``work/``."""
        return Idea2VideoPipeline(
            chat_model=self.chat,
            image_generator=self.image,
            video_generator=self.video,
            working_dir=str(self.project_dir / "work"),
            num_image_candidates=self.cfg.num_image_candidates,
        )

    def scene(self, i: int) -> Script2VideoPipeline:
        """The per-scene pipeline, working in ``work/scene_<i>/``."""
        if not isinstance(i, int) or isinstance(i, bool) or i < 0:
            raise ValueError(f"bad scene index: {i!r}")
        working_dir = self.project_dir / "work" / f"scene_{i}"
        working_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return Script2VideoPipeline(
            chat_model=self.chat,
            image_generator=self.image,
            video_generator=self.video,
            working_dir=str(working_dir),
            num_image_candidates=self.cfg.num_image_candidates,
        )

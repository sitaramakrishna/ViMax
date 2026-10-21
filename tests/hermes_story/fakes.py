"""Fake providers for hermes_story tests.  None of them touch the network."""

from __future__ import annotations

import json

from PIL import Image
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from pydantic import Field

from interfaces.image_output import ImageOutput
from interfaces.video_output import VideoOutput

from hermes_story import guarded


class FakeImageGen:
    """Returns a tiny landscape PIL image and records every call."""

    def __init__(self, raise_exc: BaseException | None = None, log_path=None) -> None:
        self.raise_exc = raise_exc
        self.log_path = log_path
        #: Optional ``(prompt, refs) -> bool`` hook to fail selected calls.
        self.fail_on = None
        self.calls: list[dict] = []

    async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
        refs = list(reference_image_paths or [])
        self.calls.append(
            {
                "prompt": prompt,
                "reference_image_paths": refs,
                "refs": [str(ref) for ref in refs],
                **kwargs,
            }
        )
        if self.log_path is not None:
            with open(self.log_path, "a", encoding="utf-8") as log:
                log.write(json.dumps({"prompt": prompt, "refs": [str(r) for r in refs]}) + "\n")
        if self.fail_on is not None and self.fail_on(prompt, [str(ref) for ref in refs]):
            raise RuntimeError("fake image failure")
        if self.raise_exc is not None:
            raise self.raise_exc
        return ImageOutput(fmt="pil", ext="png", data=Image.new("RGB", (16, 9), (200, 30, 30)))


def image_log_path(pid: str):
    """Where subprocess fake image generators append one JSON line per call."""
    from hermes_story import project

    return project.path(pid) / "fake-image-log.jsonl"

def fake_log_has_reference(pid: str, needle: str) -> bool:
    """True when a logged fake image call used a reference path containing ``needle``."""
    path = image_log_path(pid)
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if any(needle in ref for ref in json.loads(line)["refs"]):
            return True
    return False

class FakeVideoGen:
    """Returns an ``mp4`` byte output and records every call.

    ``item`` is read from :data:`hermes_story.guarded.CURRENT_ITEM`, and
    ``max_concurrent`` tracks how many calls were ever in flight at once.
    """

    def __init__(self, raise_exc: BaseException | None = None) -> None:
        self.raise_exc = raise_exc
        self.calls: list[dict] = []
        self.in_flight = 0
        self.max_concurrent = 0

    async def generate_single_video(self, prompt="", reference_image_paths=None, **kwargs):
        self.calls.append(
            {
                "item": guarded.current_item(),
                "prompt": prompt,
                "reference_image_paths": list(reference_image_paths or []),
                **kwargs,
            }
        )
        self.in_flight += 1
        self.max_concurrent = max(self.max_concurrent, self.in_flight)
        try:
            if self.raise_exc is not None:
                raise self.raise_exc
            return VideoOutput(fmt="bytes", ext="mp4", data=b"mp4")
        finally:
            self.in_flight -= 1


class FakeChat(FakeListChatModel):
    """A queued-response chat model usable with ``chat_model | parser``."""

    #: The text of every prompt handed to the model, oldest first.
    calls: list = Field(default_factory=list)

    def __init__(self, responses: list[str] | None = None, **kwargs) -> None:
        super().__init__(responses=list(responses or ["ok"]), **kwargs)

    def queue(self, response: str) -> None:
        """Append ``response``; the next call returns it.

        ``FakeListChatModel`` cycles back to the first response after the last
        one, so point ``i`` at the newly queued reply.
        """
        self.responses.append(response)
        self.i = len(self.responses) - 1

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(_messages_text(messages))
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)


def _messages_text(messages) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", message)
        parts.append(content if isinstance(content, str) else json.dumps(content))
    return "\n".join(parts)


class FakeFailChat(FakeChat):
    """Raises ``message`` on every call, for the CLI redaction test."""

    message: str = "fake failure"

    def __init__(self, message: str = "fake failure", **kwargs) -> None:
        super().__init__(responses=[], message=message, **kwargs)

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError(self.message)


class FakeStageChat(FakeChat):
    """``FakeChat`` that recognises the image-stage prompts by their schema.

    The frames stage asks for a camera tree, a reference selection and a best
    candidate; which of the last two happens depends on how many image
    candidates survive, so the reply is chosen from the prompt rather than a
    fixed position in ``responses``.
    """

    def _call(self, messages, stop=None, run_manager=None, **kwargs):
        text = _messages_text(messages)
        if "camera_parent_items" in text:
            self.calls.append(text)
            return canned.CAMERA_TREE
        if "ref_image_indices" in text:
            self.calls.append(text)
            return canned.REFERENCE_SELECTION
        if "best_image_index" in text:
            self.calls.append(text)
            return canned.BEST_IMAGE
        return super()._call(messages, stop=stop, run_manager=run_manager, **kwargs)


class canned:
    """Canned model replies in the exact shapes ViMax's parsers expect.

    The film is fixed so later tests can rely on it: two scenes, two visible
    characters (``Ana`` and ``Bolt``), and one shot per scene.
    """

    STORY = (
        "A robot learns to paint.\n\n"
        "Scene 1: Ana switches on a rusty robot in her studio.\n"
        "Scene 2: Bolt finishes its first painting and shows Ana."
    )

    SCENE_0 = (
        "INT. STUDIO - DAY\n\n"
        "ANA (30s, paint-stained apron) wipes dust off an old robot. "
        "She flips its switch and steps back."
    )
    SCENE_1 = (
        "INT. STUDIO - LATER\n\n"
        "BOLT (a boxy rusted robot) holds up a canvas covered in blue strokes. "
        "Ana claps."
    )

    SCRIPT = json.dumps({"script": [SCENE_0, SCENE_1]})

    CHARACTERS = json.dumps(
        {
            "characters": [
                {
                    "idx": 0,
                    "identifier_in_scene": "Ana",
                    "is_visible": True,
                    "static_features": "Ana has short dark hair, brown eyes and a lean build.",
                    "dynamic_features": "Wearing a paint-stained denim apron over a grey shirt.",
                },
                {
                    "idx": 1,
                    "identifier_in_scene": "Bolt",
                    "is_visible": True,
                    "static_features": "Bolt is a boxy rusted robot with round amber eyes.",
                    "dynamic_features": "Wearing a chipped blue shoulder plate.",
                },
            ]
        }
    )

    STORYBOARD_0 = json.dumps(
        {
            "storyboard": [
                {
                    "idx": 0,
                    "is_last": True,
                    "cam_idx": 0,
                    "visual_desc": (
                        "Wide shot of the cluttered studio at eye level. "
                        "<Ana> stands on the left, facing right, one hand on the "
                        "robot's dusty shoulder. Cool window light falls across "
                        "the workbench."
                    ),
                    "audio_desc": "[Sound Effect] Dust cloth brushing metal.",
                }
            ]
        }
    )

    STORYBOARD_1 = json.dumps(
        {
            "storyboard": [
                {
                    "idx": 0,
                    "is_last": True,
                    "cam_idx": 0,
                    "visual_desc": (
                        "Medium shot at eye level. <Bolt> stands on the right, "
                        "facing the camera, holding up a blue canvas. "
                        "The studio wall is softly out of focus behind it."
                    ),
                    "audio_desc": "[Speaker] Ana (happy): You did it, Bolt.",
                }
            ]
        }
    )

    DECOMP_0 = json.dumps(
        {
            "ff_desc": (
                "Wide shot of the studio at eye level. Ana (short dark hair, "
                "paint-stained apron) stands on the left, facing right, one "
                "hand on a dusty robot's shoulder."
            ),
            "ff_vis_char_idxs": [0],
            "lf_desc": (
                "Wide shot of the studio. Ana has stepped back from the robot, "
                "still facing right, her hands at her sides."
            ),
            "lf_vis_char_idxs": [0],
            "motion_desc": (
                "Static camera. The woman in the paint-stained apron lifts her "
                "hand from the robot and takes one step backwards."
            ),
            "variation_type": "small",
            "variation_reason": "Only the character's pose changes within the same framing.",
        }
    )

    DECOMP_1 = json.dumps(
        {
            "ff_desc": (
                "Medium shot at eye level. Bolt (boxy rusted robot with amber "
                "eyes) stands on the right facing the camera, canvas lowered."
            ),
            "ff_vis_char_idxs": [1],
            "lf_desc": (
                "Medium shot at eye level. Bolt holds the blue canvas up at "
                "chest height, facing the camera, its amber eyes bright."
            ),
            "lf_vis_char_idxs": [1],
            "motion_desc": (
                "Static camera. The boxy rusted robot raises the canvas from "
                "waist height to chest height."
            ),
            "variation_type": "medium",
            "variation_reason": "The robot turns from the back to face the camera.",
        }
    )

    #: Responses in the order the text stages ask for them.
    TEXT_FLOW = [STORY, SCRIPT, CHARACTERS, STORYBOARD_0, STORYBOARD_1, DECOMP_0, DECOMP_1]

    #: Camera tree, first-frame prompt and image choice for the frames stage.
    CAMERA_TREE = json.dumps({"camera_parent_items": [None]})

    #: Reference selection: keep the first (front portrait) reference.
    REFERENCE_SELECTION = json.dumps(
        {
            "ref_image_indices": [0],
            "text_prompt": "Keep the character's face from Image 0.",
        }
    )

    #: The first candidate is always the winner.
    BEST_IMAGE = json.dumps(
        {"best_image_index": 0, "reason": "The first candidate is the most consistent."}
    )

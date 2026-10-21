"""Generators that charge the film ledger around every paid call."""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable

from hermes_story.ledger import Ledger

#: The item currently being produced.  Stage code sets this with :func:`item`.
CURRENT_ITEM: contextvars.ContextVar[str] = contextvars.ContextVar(
    "hermes_story_current_item", default=""
)


def current_item() -> str:
    """Return the item id set by the surrounding :func:`item` block."""
    return CURRENT_ITEM.get()


@contextlib.contextmanager
def item(item_id: str):
    """Mark ``item_id`` as the item being produced."""
    token = CURRENT_ITEM.set(item_id)
    try:
        yield
    finally:
        CURRENT_ITEM.reset(token)


class _Guarded:
    """Shared reserve -> call -> settle flow for both generators."""

    def __init__(self, inner, ledger: Ledger, model: str, item_fn: Callable[[], str]) -> None:
        self.inner = inner
        self.ledger = ledger
        self.model = model
        self.item_fn = item_fn

    async def _settle(self, kind: str, est: float, method, prompt, references, kwargs):
        item_id = self.item_fn()
        entry_id = self.ledger.reserve(kind, item_id, est, model=self.model)
        try:
            result = await method(prompt, reference_image_paths=references, **kwargs)
        except (ValueError, TypeError):
            # Bad arguments: nothing was sent, so the reservation is free.
            self.ledger.release(entry_id)
            raise
        except BaseException:
            # Conservative: any other failure may have reached the provider.
            self.ledger.uncertain(entry_id)
            raise
        self.ledger.commit(entry_id, est)
        return result


class GuardedImageGenerator(_Guarded):
    """An image generator that reserves and charges a flat price per call."""

    def __init__(
        self,
        inner,
        ledger: Ledger,
        price: float,
        model: str,
        item_fn: Callable[[], str] = current_item,
    ) -> None:
        super().__init__(inner, ledger, model, item_fn)
        self.price = float(price)

    async def generate_single_image(self, prompt, reference_image_paths=None, **kwargs):
        return await self._settle(
            "image", self.price, self.inner.generate_single_image, prompt, reference_image_paths, kwargs
        )


class GuardedVideoGenerator(_Guarded):
    """A video generator that charges per second of requested duration."""

    def __init__(
        self,
        inner,
        ledger: Ledger,
        price_per_s: float,
        duration_s: int,
        model: str,
        item_fn: Callable[[], str] = current_item,
    ) -> None:
        super().__init__(inner, ledger, model, item_fn)
        self.price_per_s = float(price_per_s)
        self.duration_s = int(duration_s)

    @property
    def price(self) -> float:
        return self.price_per_s * self.duration_s

    async def generate_single_video(self, prompt, reference_image_paths=None, **kwargs):
        return await self._settle(
            "video", self.price, self.inner.generate_single_video, prompt, reference_image_paths, kwargs
        )

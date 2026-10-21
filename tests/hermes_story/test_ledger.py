import asyncio, pytest
from hermes_story import project, ledger, guarded
from tests.hermes_story.fakes import FakeImageGen, FakeVideoGen

@pytest.fixture
def pid(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_STORY_ROOT", str(tmp_path))
    project.create("20261021-0905-demo", {})
    return "20261021-0905-demo"

def test_cap_blocks_before_call(pid):
    led = ledger.Ledger(pid, cap_usd=0.10)
    inner = FakeImageGen()
    g = guarded.GuardedImageGenerator(inner, led, price=0.04, model="m", item_fn=lambda: "sheet:Ana")
    async def run():
        await g.generate_single_image("a"); await g.generate_single_image("b")
        with pytest.raises(ledger.CapExceeded):
            await g.generate_single_image("c")
    asyncio.run(run())
    assert len(inner.calls) == 2
    assert led.spent() == pytest.approx(0.08)

def test_video_cost_uses_duration(pid):
    led = ledger.Ledger(pid, cap_usd=10)
    g = guarded.GuardedVideoGenerator(FakeVideoGen(), led, price_per_s=0.03, duration_s=8, model="v", item_fn=lambda: "clip:0.0")
    asyncio.run(g.generate_single_video("p", ["f.png"]))
    assert led.spent() == pytest.approx(0.24)

def test_timeout_after_send_is_uncertain(pid):
    led = ledger.Ledger(pid, cap_usd=10)
    g = guarded.GuardedVideoGenerator(FakeVideoGen(raise_exc=asyncio.TimeoutError()), led, 0.03, 8, "v", lambda: "clip:0.0")
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(g.generate_single_video("p", []))
    assert led.spent() == pytest.approx(0.24)
    assert ledger.Ledger(pid, 10).entries()[-1]["status"] == "uncertain"

def test_value_error_is_released(pid):
    led = ledger.Ledger(pid, cap_usd=10)
    g = guarded.GuardedImageGenerator(FakeImageGen(raise_exc=ValueError("bad")), led, 0.04, "m", lambda: "x")
    with pytest.raises(ValueError):
        asyncio.run(g.generate_single_image("p"))
    assert led.spent() == 0

def test_ledger_survives_reload(pid):
    led = ledger.Ledger(pid, cap_usd=10)
    e = led.reserve("image", "sheet:Ana", 0.04); led.commit(e, 0.04)
    assert ledger.Ledger(pid, 10).spent() == pytest.approx(0.04)


def test_cap_alias(pid):
    led = ledger.Ledger(pid, cap_usd=1)
    led.cap = 2
    assert led.cap_usd == 2 and led.cap == 2


def test_stale_reservation_from_dead_process_is_charged(pid):
    import json
    led = ledger.Ledger(pid, cap_usd=10)
    led.reserve("video", "clip:0.0", 0.24)
    p = led.path
    data = json.loads(p.read_text())
    data["entries"][-1]["proc"] = -1          # pretend another process left it
    p.write_text(json.dumps(data))
    led2 = ledger.Ledger(pid, cap_usd=10)
    assert led2.entries()[-1]["status"] == "uncertain"
    assert led2.spent() == pytest.approx(0.24)

# Unit tests for the FORCE_THINKING transforms in proxy.py (pure functions; no server).
# Run from the router dir with its venv:
#   cd ~/ai/qwen36/router && .venv/bin/python -m pytest tests/test_force_thinking.py -q
#   (or: .venv/bin/python tests/test_force_thinking.py)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import proxy.py
import proxy


def ft(payload):
    p = dict(payload); tags = proxy._force_thinking(p); return p, tags


def test_ctk_enable_thinking_false_to_true():
    p, tags = ft({"chat_template_kwargs": {"enable_thinking": False}})
    assert p["chat_template_kwargs"]["enable_thinking"] is True
    assert any("enable_thinking false->true" in t for t in tags)


def test_toplevel_enable_thinking_false_dropped():
    p, tags = ft({"enable_thinking": False, "messages": []})
    assert "enable_thinking" not in p
    assert any("top-level enable_thinking:false dropped" in t for t in tags)


def test_reasoning_effort_none_to_low_toplevel():
    p, tags = ft({"reasoning_effort": "none"})
    assert p["reasoning_effort"] == "low"
    assert any("reasoning_effort 'none'->low" in t for t in tags)


def test_reasoning_effort_in_kwargs_to_low():
    p, tags = ft({"chat_template_kwargs": {"reasoning_effort": "off"}})
    assert p["chat_template_kwargs"]["reasoning_effort"] == "low"


def test_reasoning_effort_valid_untouched():
    for v in ("low", "medium", "high", "xhigh"):
        p, tags = ft({"reasoning_effort": v})
        assert p["reasoning_effort"] == v
        assert tags == []


def test_think_off_stripped_system_and_user():
    p, tags = ft({"messages": [
        {"role": "system", "content": "Score this.<|think_off|>"},
        {"role": "user", "content": "Item A <|think_off|>"},
    ]})
    assert "<|think_off|>" not in p["messages"][0]["content"]
    assert "<|think_off|>" not in p["messages"][1]["content"]
    assert p["messages"][0]["content"] == "Score this."
    assert any("stripped <|think_off|>" in t for t in tags)


def test_think_off_in_list_content_parts():
    p, tags = ft({"messages": [
        {"role": "user", "content": [{"type": "text", "text": "hi <|think_off|>"}]},
    ]})
    assert p["messages"][0]["content"][0]["text"] == "hi "
    assert any("stripped <|think_off|>" in t for t in tags)


def test_already_thinking_on_untouched():
    # explicit enable_thinking true -> no flip, no mutation
    p, tags = ft({"chat_template_kwargs": {"enable_thinking": True}, "messages": [{"role": "user", "content": "hi"}]})
    assert tags == [] and p["chat_template_kwargs"]["enable_thinking"] is True
    # nothing thinking-related -> no flip
    p, tags = ft({"messages": [{"role": "user", "content": "hi"}], "temperature": 0.7})
    assert tags == [] and "chat_template_kwargs" not in p


def test_force_thinking_off_bypass():
    saved = proxy.FORCE_THINKING
    try:
        proxy.FORCE_THINKING = False
        p, tags = ft({"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "none"})
        assert tags == []
        assert p["chat_template_kwargs"]["enable_thinking"] is False  # untouched
        assert p["reasoning_effort"] == "none"
    finally:
        proxy.FORCE_THINKING = saved


def test_floor_raises_small():
    p = {"max_tokens": 256}
    r = proxy._floor_max_tokens(p, 512)
    assert p["max_tokens"] == 512 and r == ("max_tokens", 256, 512)


def test_floor_never_lowers():
    p = {"max_tokens": 8000}
    r = proxy._floor_max_tokens(p, 512)
    assert p["max_tokens"] == 8000 and r is None


def test_floor_absent_untouched():
    p = {"temperature": 0.7}
    r = proxy._floor_max_tokens(p, 4096)
    assert "max_tokens" not in p and r is None


def test_floor_flashnext_4096():
    p = {"max_tokens": 700}
    r = proxy._floor_max_tokens(p, 4096)   # flash-next floor
    assert p["max_tokens"] == 4096


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} tests passed")

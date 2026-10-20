# Unit tests for the NO_OUTPUT_CAPS cap-strip, the X-Respect-Max-Tokens opt-out, and the split read
# timeouts in proxy.py (pure functions / module constants; no server).
# Run from the router dir with its venv:
#   cd ~/ai/qwen36/router && .venv/bin/python -m pytest tests/test_output_caps.py -q
#   (or: .venv/bin/python tests/test_output_caps.py)
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import proxy.py
import proxy


# ---- cap strip ----

def test_strip_removes_all_three_keys():
    p = {"model": "qwen36", "max_tokens": 50, "max_completion_tokens": 60, "n_predict": 70, "temperature": 0}
    removed = proxy._strip_output_caps(p)
    assert ("max_tokens", 50) in removed
    assert ("max_completion_tokens", 60) in removed
    assert ("n_predict", 70) in removed
    assert all(k not in p for k in ("max_tokens", "max_completion_tokens", "n_predict"))
    assert p["temperature"] == 0 and p["model"] == "qwen36"   # non-cap keys untouched


def test_strip_noop_when_absent():
    p = {"model": "qwen36", "messages": []}
    removed = proxy._strip_output_caps(p)
    assert removed == [] and p == {"model": "qwen36", "messages": []}


def test_cap_keys_are_the_three_budget_keys():
    assert set(proxy._CAP_KEYS) == {"max_tokens", "max_completion_tokens", "n_predict"}


# ---- respect-caps decision (NO_OUTPUT_CAPS on) ----

def test_respect_false_by_default_when_caps_on():
    saved = proxy.NO_OUTPUT_CAPS
    try:
        proxy.NO_OUTPUT_CAPS = True
        assert proxy._respect_caps(None) is False      # no header -> strip
        assert proxy._respect_caps("") is False
        assert proxy._respect_caps("0") is False
        assert proxy._respect_caps("yes") is False     # only "1" opts out
    finally:
        proxy.NO_OUTPUT_CAPS = saved


def test_respect_true_with_opt_out_header():
    saved = proxy.NO_OUTPUT_CAPS
    try:
        proxy.NO_OUTPUT_CAPS = True
        assert proxy._respect_caps("1") is True
        assert proxy._respect_caps(" 1 ") is True      # trimmed
    finally:
        proxy.NO_OUTPUT_CAPS = saved


def test_respect_always_true_when_caps_disabled_globally():
    saved = proxy.NO_OUTPUT_CAPS
    try:
        proxy.NO_OUTPUT_CAPS = False
        assert proxy._respect_caps(None) is True       # feature off -> caps respected regardless
        assert proxy._respect_caps("0") is True
    finally:
        proxy.NO_OUTPUT_CAPS = saved


# ---- read timeouts (streaming per-chunk vs non-streaming whole-body) ----

def test_timeouts_default_values():
    assert proxy.STREAM_READ_TIMEOUT == 900.0
    assert proxy.NONSTREAM_READ_TIMEOUT == 7200.0


def test_nonstream_timeout_exceeds_stream():
    assert proxy.NONSTREAM_READ_TIMEOUT > proxy.STREAM_READ_TIMEOUT


def test_read_timeout_for_picks_by_streaming():
    assert proxy._read_timeout_for(True) == proxy.STREAM_READ_TIMEOUT      # per-chunk for streams
    assert proxy._read_timeout_for(False) == proxy.NONSTREAM_READ_TIMEOUT  # whole-body for non-stream


def test_parse_timeout_env_and_zero():
    os.environ["_T_TEST_A"] = "1234"
    os.environ["_T_TEST_Z"] = "0"
    try:
        assert proxy._parse_timeout("_T_TEST_A", "900") == 1234.0
        assert proxy._parse_timeout("_T_TEST_Z", "900") is None   # 0 -> unbounded
        assert proxy._parse_timeout("_T_TEST_MISSING", "7200") == 7200.0
    finally:
        del os.environ["_T_TEST_A"], os.environ["_T_TEST_Z"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} tests passed")

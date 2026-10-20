# Proposal (NOT applied): router non-thinking sampling profile for the 3.8 entries

**What:** when a request to a **Qwen3.8 entry** (27B, flash-next, flash-next-batch) **explicitly disables
thinking** *and* carries **no sampling params of its own**, inject the Qwen3.8 card's **non-thinking** profile:
`temperature 0.7, top_p 0.80, top_k 20, min_p 0, presence_penalty 1.5`.

**Guarantees:**
- **Never overrides a client-sent value.** It fires only when the request has *zero* sampling keys; and it
  uses `setdefault`, so even a stray key is never clobbered.
- **Scoped to the 3.8 line** (by the client-sent id/alias, exactly like `TOOL_MIN_BY_MODEL`). The **35B is
  Qwen3.6 and is deliberately excluded** — it has its own card (and its launcher already sets temp 0.6).
- **Thinking ON (the default) → untouched.** Only an *explicit* `enable_thinking=false` triggers it.
- Env-overridable (`NONTHINKING_SAMPLING`, `NONTHINKING_MODELS`) like the other router knobs.

**Why a router default at all:** llama-server's own defaults for these entries are the *thinking* profile
(temp 1.0 / top_p 0.95 — from the launcher flags + GGUF `general.sampling.*`). A client that turns thinking
off but sends no sampling params would otherwise decode a non-thinking answer with thinking-tuned sampling
(too hot, no presence penalty), which the card warns against. This closes that gap without affecting any
client that sets its own sampling.

**Known gap (documented, not handled):** detects `enable_thinking=false` via `chat_template_kwargs` or a
top-level key; it does **not** detect the `/no_think` soft-switch embedded in message content. Add that only
if a client actually uses it.

---

## Diff (against `router/proxy.py`)

```diff
@@ -50,6 +50,25 @@ TOOL_MIN_BY_MODEL = json.loads(os.environ.get("TOOL_MIN_BY_MODEL", json.dumps({
     "qwen38-next-batch": 4096,
 })))
 
+# Qwen3.8 NON-THINKING sampling profile (from the model card). Injected ONLY when a 3.8 entry request
+# (a) EXPLICITLY disables thinking AND (b) carries NO sampling params of its own — never overriding a
+# client value. Scoped by the client-sent id/alias (like TOOL_MIN_BY_MODEL). The 35B is Qwen3.6 (its own
+# card / temp 0.6) and is intentionally absent. All env-overridable.
+NONTHINKING_PROFILE = json.loads(os.environ.get("NONTHINKING_SAMPLING", json.dumps(
+    {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5})))
+NONTHINKING_MODELS = set(json.loads(os.environ.get("NONTHINKING_MODELS", json.dumps([
+    "qwen-38-27b", "qwen38", "qwen38-27b", "qwen36", "qwen36-q6", "qwen36-text",
+    "qwen-38-flash-next", "flash-next", "qwen38-next",
+    "qwen-38-flash-next-batch", "flash-next-batch", "qwen38-next-batch",
+]))))
+# Any of these present in the body => the client chose its own sampling; we leave the whole request alone.
+_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "typical_p", "presence_penalty",
+                  "frequency_penalty", "repeat_penalty", "top_n_sigma", "xtc_probability",
+                  "xtc_threshold", "dynatemp_range", "mirostat", "dry_multiplier", "dry_base")
+
 # Read timeout: generous, but FINITE.
 #
 # This was read=None, for a good reason - a cold model swap pages ~21 GB off disk and can take
@@ -97,6 +116,33 @@ def _maybe_inject_system(payload: dict) -> None:
     if not (msgs and msgs[0].get("role") in ("system", "developer")):
         msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
 
+def _thinking_disabled(payload: dict) -> bool:
+    """True only if the client EXPLICITLY turned thinking off (chat_template_kwargs.enable_thinking=false,
+    or a top-level enable_thinking=false 'equivalent'). Absent => thinking on (default) => False."""
+    ctk = payload.get("chat_template_kwargs")
+    if isinstance(ctk, dict) and ctk.get("enable_thinking") is False:
+        return True
+    return payload.get("enable_thinking") is False
+
+def _maybe_inject_nonthinking_sampling(payload: dict):
+    """3.8 entry + thinking explicitly off + NO client sampling params => apply the 3.8 non-thinking card
+    profile. Returns the injected dict (for logging) or None. Never overrides a client-sent value."""
+    if payload.get("model") not in NONTHINKING_MODELS:
+        return None
+    if not _thinking_disabled(payload):
+        return None
+    if any(k in payload for k in _SAMPLING_KEYS):   # client sent sampling -> leave the request untouched
+        return None
+    for k, v in NONTHINKING_PROFILE.items():
+        payload.setdefault(k, v)                    # belt-and-suspenders: never clobber
+    return dict(NONTHINKING_PROFILE)
+
 
 async def _forward(request, body: bytes, streaming_hint: bool):
     url = SWAP_BACKEND + request.url.path
@@ -137,6 +183,10 @@ async def chat(request):
     if payload:
         _maybe_inject_system(payload)
         bumped = _guard_tool_budget(payload)
         if bumped:
             print(f"[router] tools present: raised {bumped[0]} {bumped[1]} -> {bumped[2]} "
                   f"(avoid tool-call truncation)", flush=True)
+        inj = _maybe_inject_nonthinking_sampling(payload)
+        if inj:
+            print(f"[router] non-thinking + no client sampling on {payload.get('model')}: "
+                  f"injected 3.8 non-thinking profile {inj}", flush=True)
         body = json.dumps(payload).encode()   # `model` passes through untouched — it is the swap key
```

---

## Tests (`router/tests/test_nonthinking_sampling.py`) — pure-function, no server

```python
# Run: cd ~/ai/qwen36 && python3 -m pytest router/tests/test_nonthinking_sampling.py -q
# (or: python3 router/tests/test_nonthinking_sampling.py)
from router.proxy import _maybe_inject_nonthinking_sampling as inject, NONTHINKING_PROFILE

NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

def case(payload):
    p = dict(payload); r = inject(p); return p, r

def test_injects_when_38_nothink_and_no_sampling():
    p, r = case({"model": "qwen-38-flash-next", **NOTHINK})
    assert r == NONTHINKING_PROFILE
    for k, v in NONTHINKING_PROFILE.items():
        assert p[k] == v                       # full profile applied

def test_never_overrides_a_client_param():
    # client sent temperature -> leave the ENTIRE request alone (no partial fill, no override)
    p, r = case({"model": "qwen-38-27b", "temperature": 0.2, **NOTHINK})
    assert r is None
    assert p["temperature"] == 0.2
    assert "top_p" not in p and "presence_penalty" not in p

def test_thinking_on_is_untouched():
    p, r = case({"model": "qwen-38-flash-next"})           # no enable_thinking => thinking on
    assert r is None and "temperature" not in p

def test_explicit_enable_thinking_true_is_untouched():
    p, r = case({"model": "qwen-38-27b", "chat_template_kwargs": {"enable_thinking": True}})
    assert r is None and "temperature" not in p

def test_35b_is_out_of_scope():
    p, r = case({"model": "qwen-35b", **NOTHINK})
    assert r is None and "temperature" not in p
    p, r = case({"model": "qwen-35b-batch", **NOTHINK})
    assert r is None and "temperature" not in p

def test_top_level_enable_thinking_false_equivalent():
    p, r = case({"model": "qwen38-next", "enable_thinking": False})
    assert r == NONTHINKING_PROFILE and p["top_p"] == 0.80

def test_unknown_model_not_injected():
    p, r = case({"model": "some-other-model", **NOTHINK})
    assert r is None

def test_one_sampling_key_blocks_injection():
    # a single client sampling key (top_k) means "client chose sampling" -> no injection at all
    p, r = case({"model": "flash-next", "top_k": 50, **NOTHINK})
    assert r is None and "temperature" not in p and p["top_k"] == 50

if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for f in fns:
        f(); print("ok", f.__name__)
    print(f"\n{len(fns)} tests passed")
```

**Live coverage:** `regress.sh #13` already exercises the router's body-rewrite path end-to-end; add a
non-thinking case there once this lands so a real `enable_thinking=false` request is checked through :8000.

#!/usr/bin/env python3
"""
Qwen3.6 router — one OpenAI-compatible endpoint on :8000.

  client -> router :8000 (tool-budget guard, system-prompt injection) -> llama-swap :9000 -> llama-server

WHY THIS ROUTER EXISTS (vision is NOT retired — the 27B and flash-next are multimodal):
Model routing, image requests included, is llama-swap's job: we pass `model` through untouched (it is
the swap key), so a multimodal model receives image content unchanged. The router's own job is the two
things llama-swap CANNOT express: the tool-budget guard and system-prompt injection. llama-swap's
`filters.setParams` is an unconditional set ("always runs for the model"), not a floor, and it
cannot condition on `tools` being present. Configuring max_tokens=512 there would cap EVERY
request at 512 tokens, truncating long code generation and 100K RAG answers. The guard needs
    IF tools present AND max_tokens < floor THEN raise to floor
which is conditional logic on the request body. Hence the chain. See regression test #13.

Aliases ("qwen36"/"qwen38" -> qwen-38-27b, "qwen36-35b" -> qwen-35b) are configured in swap/config.yaml.

Deps: starlette, uvicorn, httpx (installed into router/.venv by setup.sh via uv).
"""
import os, json, time
import httpx
from starlette.applications import Starlette
from starlette.responses import Response, StreamingResponse, JSONResponse
from starlette.routing import Route

SWAP_BACKEND  = os.environ.get("SWAP_BACKEND", "http://127.0.0.1:9000")
INJECT_SYSTEM = os.environ.get("INJECT_SYSTEM", "1") == "1"
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
# Tool-call turns need room for thinking + the tool JSON. Too small a budget truncates the
# tool call mid-argument -> llama.cpp 500 ("Failed to parse tool call arguments as JSON").
# Root-caused from the M4 soak (10/806 errors). Raise max_tokens to a floor when tools are present.
# Set TOOL_MIN_TOKENS=0 to disable the guard.
TOOL_MIN_TOKENS = int(os.environ.get("TOOL_MIN_TOKENS", "512"))

# PER-MODEL FLOORS. A single global floor cannot serve models whose reasoning lengths differ by 4x.
# The 512 default was tuned on the 27B/35B (reasoning ~1500 chars). The 122B thinks LONG and
# VARIABLY -- measured 4472 chars one run and 6647 the next on the SAME prompt; at max_tokens=2048
# it intermittently emitted its whole budget into reasoning_content and returned EMPTY content
# (M2 gate 5). 512 would truncate its tool calls outright. Keyed by the model/alias the client
# sends; falls back to TOOL_MIN_TOKENS.
TOOL_MIN_BY_MODEL = json.loads(os.environ.get("TOOL_MIN_BY_MODEL", json.dumps({
    # qwen4exp flash-next thinks long and variably -> floor its tool budget to 4096 (keyed by id + aliases)
    "qwen-38-flash-next": 4096,
    "flash-next":  4096,
    "qwen38-next": 4096,
    "qwen-38-flash-next-batch": 4096,
    "flash-next-batch": 4096,
    "qwen38-next-batch": 4096,
})))

# FORCE THINKING (default ON). Thinking is always on for chat requests; clients CANNOT disable it.
# This supersedes router/NONTHINKING-SAMPLING.proposal.md (which would have injected a non-thinking
# sampling profile — now moot, since the non-thinking path never reaches the model). Set FORCE_THINKING=0
# to restore pass-through. The three neutralized disable mechanisms (from the chat-template audit):
#   - chat_template_kwargs.enable_thinking=false (all entries honour it)   -> set true
#   - a top-level enable_thinking=false (non-standard "equivalent")        -> dropped
#   - reasoning_effort set to a non-standard value e.g. "none" (the 3.8 template RAISES on unknown values;
#     the 3.6/35B template ignores reasoning_effort)                        -> remapped to "low"
#   - the <|think_off|> soft switch (the 3.6/35B template honours it in SYSTEM or message content; the 3.8
#     templates treat it as literal text)                                   -> stripped from every message
FORCE_THINKING = os.environ.get("FORCE_THINKING", "1").strip().lower() not in ("0", "false", "no", "off", "")
_REFFORT_KEEP = {"low", "medium", "high", "xhigh"}   # thinking-ON effort levels the 3.8 template accepts
_THINK_OFF = "<|think_off|>"

# Read timeout: generous, but FINITE.
#
# This was read=None, for a good reason - a cold model swap pages ~21 GB off disk and can take
# ~25s before the first byte, and long-context generations run for minutes. Neither should time
# out at the proxy.
#
# But None has a failure mode that only shows up under load. llama-server queues requests beyond
# its slot count, and a queued request produces no bytes while it waits. With no read deadline a
# saturated backend does not return errors, it returns nothing, forever: the client blocks, the
# caller has no signal, and the only symptom is a job that never finishes. A finite deadline
# turns that silent hang into a timeout the caller can retry or report.
#
# 900s is chosen to sit above the worst legitimate wait (a cold 122B swap plus a long generation)
# and below "forever". Override with ROUTER_READ_TIMEOUT if a workload genuinely needs longer;
# set it to 0 to restore the old unbounded behaviour.
_read_timeout_env = os.environ.get("ROUTER_READ_TIMEOUT", "900").strip()
READ_TIMEOUT = None if _read_timeout_env in ("0", "none", "") else float(_read_timeout_env)
CLIENT = httpx.AsyncClient(
    timeout=httpx.Timeout(connect=10.0, read=READ_TIMEOUT, write=60.0, pool=None)
)


def _guard_tool_budget(payload: dict):
    """If tools are present, ensure the output budget can't truncate the tool call mid-argument.
    The floor is PER-MODEL: verbose models (the 122B) need a much larger one than the 27B/35B."""
    floor = TOOL_MIN_BY_MODEL.get(payload.get("model"), TOOL_MIN_TOKENS)
    if floor <= 0 or not payload.get("tools"):
        return None
    bumped = None
    for key in ("max_tokens", "max_completion_tokens"):
        v = payload.get(key)
        if isinstance(v, int) and 0 < v < floor:
            payload[key] = floor
            bumped = (key, v, floor)
    return bumped


def _maybe_inject_system(payload: dict) -> None:
    if not INJECT_SYSTEM:
        return
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return
    if not (msgs and msgs[0].get("role") in ("system", "developer")):
        msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})


def _strip_think_off(payload: dict) -> bool:
    """Remove the <|think_off|> soft switch from EVERY message's text (string content or text parts of a
    list). The 35B template honours it in system AND message content and clients put it in the system
    message, so stripping only the last user turn would not force thinking. Returns True if any was removed."""
    msgs = payload.get("messages")
    if not isinstance(msgs, list):
        return False
    hit = False
    for m in msgs:
        if not isinstance(m, dict):
            continue
        c = m.get("content")
        if isinstance(c, str) and _THINK_OFF in c:
            m["content"] = c.replace(_THINK_OFF, ""); hit = True
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and _THINK_OFF in part["text"]:
                    part["text"] = part["text"].replace(_THINK_OFF, ""); hit = True
    return hit


def _force_thinking(payload: dict) -> list:
    """Force thinking ON (behind FORCE_THINKING). Returns a list of change tags (empty => nothing flipped).
    Only neutralizes EXPLICIT disables; a request with no thinking controls is left untouched (the templates
    default to thinking-on). Never lowers max_tokens (the caller applies the per-model floor on a flip)."""
    if not FORCE_THINKING:
        return []
    changed = []
    ctk = payload.get("chat_template_kwargs")
    # 1. chat_template_kwargs.enable_thinking=false -> true
    if isinstance(ctk, dict) and ctk.get("enable_thinking") is False:
        ctk["enable_thinking"] = True
        changed.append("chat_template_kwargs.enable_thinking false->true")
    # 2. top-level enable_thinking=false -> drop (non-standard "equivalent")
    if payload.get("enable_thinking") is False:
        payload.pop("enable_thinking", None)
        changed.append("top-level enable_thinking:false dropped")
    # 3. reasoning_effort: a disabling/invalid value (e.g. "none") -> "low"; keep low/medium/high/xhigh
    def _fix_reff(d, label):
        v = d.get("reasoning_effort")
        if isinstance(v, str) and v.strip().lower() not in _REFFORT_KEEP:
            d["reasoning_effort"] = "low"; changed.append(f"{label} reasoning_effort {v!r}->low")
    _fix_reff(payload, "top-level")
    if isinstance(ctk, dict):
        _fix_reff(ctk, "chat_template_kwargs")
    # 4. <|think_off|> soft switch -> stripped everywhere
    if _strip_think_off(payload):
        changed.append("stripped <|think_off|>")
    return changed


def _floor_max_tokens(payload: dict, floor: int):
    """Raise max_tokens / max_completion_tokens to `floor` if present and below it; NEVER lower a client
    value, never add the key when absent. Returns (key, old, new) for the raised key, else None."""
    if floor <= 0:
        return None
    raised = None
    for key in ("max_tokens", "max_completion_tokens"):
        v = payload.get(key)
        if isinstance(v, int) and 0 < v < floor:
            payload[key] = floor; raised = (key, v, floor)
    return raised


async def _forward(request, body: bytes, streaming_hint: bool):
    url = SWAP_BACKEND + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    # drop hop-by-hop headers
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
    # Force identity encoding upstream. We forward resp.aiter_raw() (undecoded) and strip
    # content-encoding below, so the upstream must NOT compress. Merely dropping the client's
    # accept-encoding is not enough: httpx then auto-adds its own "gzip, deflate", the backend
    # compresses, and the client receives gzipped bytes with no content-encoding header to decode.
    headers["accept-encoding"] = "identity"
    req = CLIENT.build_request(request.method, url, headers=headers, content=body)
    resp = await CLIENT.send(req, stream=True)
    resp_headers = {k: v for k, v in resp.headers.items()
                    if k.lower() not in ("content-length", "transfer-encoding", "connection", "content-encoding")}

    ctype = resp.headers.get("content-type", "")
    if streaming_hint or "text/event-stream" in ctype:
        async def gen():
            try:
                async for chunk in resp.aiter_raw():
                    yield chunk
            finally:
                await resp.aclose()
        return StreamingResponse(gen(), status_code=resp.status_code,
                                 headers=resp_headers, media_type=ctype or "text/event-stream")
    data = await resp.aread()
    await resp.aclose()
    return Response(content=data, status_code=resp.status_code, headers=resp_headers, media_type=ctype)


async def chat(request):
    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except Exception:
        payload = {}
    if payload:
        _maybe_inject_system(payload)
        bumped = _guard_tool_budget(payload)
        if bumped:
            print(f"[router] tools present: raised {bumped[0]} {bumped[1]} -> {bumped[2]} "
                  f"(avoid tool-call truncation)", flush=True)
        flips = _force_thinking(payload)
        if flips:
            # thinking was forced on -> give it room: raise max_tokens to the per-model floor (same floors
            # as the tool guard: 512 for 27B/35B, 4096 for flash-next). Never lower a client value.
            raised = _floor_max_tokens(payload, TOOL_MIN_BY_MODEL.get(payload.get("model"), TOOL_MIN_TOKENS))
            print(f"[router] FORCE_THINKING {payload.get('model')}: {'; '.join(flips)}"
                  + (f"; raised {raised[0]} {raised[1]}->{raised[2]}" if raised
                     else "; max_tokens left as-is"), flush=True)
        body = json.dumps(payload).encode()   # `model` passes through untouched — it is the swap key
    stream = bool(payload.get("stream"))
    return await _forward(request, body, stream)


async def load_model(request):
    """Pre-warm a model WITHOUT sending a dummy chat turn. Triggers llama-swap to start (and swap
    to) the model, blocking until it is resident, then returns. Accepts model id OR alias.

      GET  /load/qwen36-q6
      POST /load            {"model": "qwen36-q6"}

    Mechanism: llama-swap starts a backend on the first request routed to it; hitting its
    /upstream/<model>/health is the lightest such request (no token generation). read=None on the
    shared client means a cold 122B load (~90s+) will not time out here."""
    model = request.path_params.get("model")
    if not model and request.method == "POST":
        try:
            model = json.loads(await request.body() or b"{}").get("model")
        except Exception:
            model = None
    if not model:
        return JSONResponse({"error": "specify a model: GET /load/<model> or POST /load {\"model\":..}"},
                            status_code=400)
    t0 = time.monotonic()
    try:
        r = await CLIENT.get(f"{SWAP_BACKEND}/upstream/{model}/health")
    except Exception as e:
        return JSONResponse({"error": f"load failed: {e.__class__.__name__}: {e}", "model": model},
                            status_code=504)
    running = []
    try:
        rr = await CLIENT.get(SWAP_BACKEND + "/running", timeout=5.0)
        running = [m.get("model") for m in rr.json().get("running", [])]
    except Exception:
        pass
    ok = r.status_code == 200
    secs = round(time.monotonic() - t0, 1)
    print(f"[router] /load {model} -> upstream {r.status_code}, running={running}, {secs}s", flush=True)
    # 404 (unknown model) is passed through as-is so callers can distinguish it from a load failure.
    status = 200 if ok else (404 if r.status_code == 404 else 502)
    return JSONResponse({"loaded": model if ok else None, "upstream_status": r.status_code,
                         "running": running, "seconds": secs}, status_code=status)


async def unload_model(request):
    """Free all GPUs (symmetric with /load). Proxies llama-swap's POST /api/models/unload.
    Evicts EVERY model, including another client's mid-job; use /unload/<model> to free just one."""
    try:
        r = await CLIENT.post(SWAP_BACKEND + "/api/models/unload", timeout=60.0)
    except Exception as e:
        return JSONResponse({"error": f"{e.__class__.__name__}: {e}"}, status_code=504)
    return JSONResponse({"unloaded": r.status_code == 200, "upstream_status": r.status_code},
                        status_code=200 if r.status_code == 200 else 502)


async def unload_one(request):
    """Free ONE model (by id or alias), leaving any other client's model resident.

      GET|POST /unload/qwen-35b   -> {"unloaded": "qwen-35b", "upstream_status": 200, "running": [...]}

    Proxies llama-swap's POST /api/models/unload/<model>. Unloading a model that is not running is a
    no-op 200; an unknown model is 404. Prefer this over /unload, which evicts EVERY model (including
    another client's mid-job)."""
    model = request.path_params["model"]
    try:
        r = await CLIENT.post(f"{SWAP_BACKEND}/api/models/unload/{model}", timeout=60.0)
    except Exception as e:
        return JSONResponse({"error": f"{e.__class__.__name__}: {e}", "model": model}, status_code=504)
    running = []
    try:
        rr = await CLIENT.get(SWAP_BACKEND + "/running", timeout=5.0)
        running = [m.get("model") for m in rr.json().get("running", [])]
    except Exception:
        pass
    status = 200 if r.status_code == 200 else (404 if r.status_code == 404 else 502)
    return JSONResponse({"unloaded": model if r.status_code == 200 else None, "upstream_status": r.status_code,
                         "running": running}, status_code=status)


async def passthrough(request):
    # /v1/models, /completion, /running, /health, etc. -> llama-swap
    body = await request.body()
    try:
        stream = bool(json.loads(body).get("stream")) if body else False
    except Exception:
        stream = False
    return await _forward(request, body, stream)


async def health(request):
    """Reports llama-swap reachability and which model is currently loaded.
    NOTE: 'no model loaded' is a NORMAL state here (ttl:0, load-on-demand) -- it is not an error,
    unlike the old two-always-resident-backends design where a down backend meant a real fault."""
    out = {}
    try:
        r = await CLIENT.get(SWAP_BACKEND + "/v1/models", timeout=5.0)
        out["swap"] = r.status_code
    except Exception as e:
        out["swap"] = f"down: {e.__class__.__name__}"
    try:
        r = await CLIENT.get(SWAP_BACKEND + "/running", timeout=5.0)
        out["running"] = [m.get("model") for m in r.json().get("running", [])] if r.status_code == 200 else r.status_code
    except Exception as e:
        out["running"] = f"unknown: {e.__class__.__name__}"
    ok = out.get("swap") == 200
    return JSONResponse({"router": "ok" if ok else "degraded", "backend": out}, status_code=200 if ok else 503)


routes = [
    Route("/healthz", health, methods=["GET"]),
    Route("/load", load_model, methods=["GET", "POST"]),
    Route("/load/{model}", load_model, methods=["GET", "POST"]),
    Route("/unload", unload_model, methods=["GET", "POST"]),
    Route("/unload/{model}", unload_one, methods=["GET", "POST"]),
    Route("/v1/chat/completions", chat, methods=["POST"]),
    Route("/{path:path}", passthrough, methods=["GET", "POST"]),   # catch-all LAST
]

app = Starlette(routes=routes)

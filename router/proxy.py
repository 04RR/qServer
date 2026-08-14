#!/usr/bin/env python3
"""
Qwen3.6 router — one OpenAI-compatible endpoint on :8000.

  client -> router :8000 (tool-budget guard, system-prompt injection) -> llama-swap :9000 -> llama-server

WHY THIS STILL EXISTS NOW THAT VISION IS RETIRED:
The image-content routing is gone (vision retired; both GPUs now serve one text model at a time),
so the only remaining job is the tool-budget guard -- and llama-swap CANNOT express it. Its
`filters.setParams` is an unconditional set ("always runs for the model"), not a floor, and it
cannot condition on `tools` being present. Configuring max_tokens=512 there would cap EVERY
request at 512 tokens, truncating long code generation and 100K RAG answers. The guard needs
    IF tools present AND max_tokens < floor THEN raise to floor
which is conditional logic on the request body. Hence the chain. See regression test #13.

Model routing is llama-swap's job: we pass `model` through untouched (it is the swap key).
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
    "qwen-122b":   4096,
    "qwen35-122b": 4096,
})))

# read=None: a cold model swap pages ~21 GB off disk and can take ~25s before the first byte;
# long-context generations run for minutes. Neither must time out at the proxy.
CLIENT = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None))


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


async def _forward(request, body: bytes, streaming_hint: bool):
    url = SWAP_BACKEND + request.url.path
    if request.url.query:
        url += "?" + request.url.query
    # drop hop-by-hop headers
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length", "connection", "accept-encoding")}
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
    """Free all GPUs (symmetric with /load). Proxies llama-swap's POST /api/models/unload."""
    try:
        r = await CLIENT.post(SWAP_BACKEND + "/api/models/unload", timeout=60.0)
    except Exception as e:
        return JSONResponse({"error": f"{e.__class__.__name__}: {e}"}, status_code=504)
    return JSONResponse({"unloaded": r.status_code == 200, "upstream_status": r.status_code},
                        status_code=200 if r.status_code == 200 else 502)


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
    Route("/v1/chat/completions", chat, methods=["POST"]),
    Route("/{path:path}", passthrough, methods=["GET", "POST"]),   # catch-all LAST
]

app = Starlette(routes=routes)

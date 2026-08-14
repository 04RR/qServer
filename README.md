# qServer — local Qwen inference stack

A bulletproof local inference stack for three Qwen models on a single dual-GPU workstation
(RTX 4090 + RTX 5060 Ti + system RAM), managed by [llama-swap](https://github.com/mostlygeek/llama-swap)
behind one OpenAI-compatible endpoint. Models are **mutually exclusive** — one resident at a time,
loaded on demand by name.

| model | id (alias) | placement | speed |
|---|---|---|---|
| 27B dense **+ vision** (Qwen3.8) Q6_K | `qwen-38-27b` (`qwen38`, `qwen36`, `qwen36-q6`) | 4090 + 5060 Ti | ~73 t/s code · 48 @100K |
| 35B-A3B MoE | `qwen-35b` (`qwen36-35b`) | 4090 + 5060 Ti | ~206 t/s |
| 122B-A10B MoE | `qwen-122b` (`qwen35-122b`) | 4090 + 5060 Ti + RAM | ~37–40 t/s |

All: 131072 context, q8_0 KV, MTP speculative decoding, thinking on. The 27B is a native
**vision-language** model (images + video) and replaced the retired 3.6-27B dense line (Q4 + Q6);
its legacy aliases (`qwen36`, `qwen36-q6`, `qwen36-text`) carry forward so existing clients keep working.

## Docs
- **[API.md](API.md)** — the OpenAI-compatible API, every endpoint, and the metrics anyone building on it needs.
- **[USAGE.md](USAGE.md)** — day-to-day commands and gotchas.
- **[RESULTS.md](RESULTS.md)** / **[LEARNINGS.md](LEARNINGS.md)** — the full build record: measurements, bugs hit, and why each decision was made.

## Layout
```
run-text-*.sh        llama-server launch scripts (one per model, every hard-won flag encoded)
router/proxy.py      OpenAI-compatible router :8000 (tool-budget guard, /load, /unload)
swap/config.yaml     llama-swap model manager :9000 (mutual-exclusion group)
systemd/             units + boot verification
templates/           chat templates (v19)
gates*.sh, regress.sh, canary.py, needle35.py, soak*.py   regression / gate suites
```

## Not included
Model weights (`models/`, `*.gguf`, ~125 GB) and the CUDA-built `llama.cpp/` are gitignored — build
llama.cpp with CUDA 12.8 and fetch the GGUFs from the [unsloth](https://huggingface.co/unsloth) repos
referenced in the run scripts.

> Requires CUDA **12.8** specifically (13.1 breaks MMQ, 13.2 produces gibberish on this arch).

# qServer — local Qwen inference stack

A bulletproof local inference stack for several Qwen models on a single dual-GPU workstation
(RTX 4090 + RTX 5060 Ti + system RAM), managed by [llama-swap](https://github.com/mostlygeek/llama-swap)
behind one OpenAI-compatible endpoint. Models are **mutually exclusive** — one resident at a time,
loaded on demand by name.

| model | id (alias) | placement | speed |
|---|---|---|---|
| 27B dense **+ vision** (Qwen3.8), 2 quant profiles | `qwen-38-27b` (`qwen38`, `qwen36`, `qwen36-q6`) | 4090 + 5060 Ti | UD-Q6_K_M ~69 code · 57 RAG@32K · plain Q6_K ~+8% |
| 35B-A3B MoE | `qwen-35b` (`qwen36-35b`) | 4090 + 5060 Ti | ~206 t/s |
| 125B-A6B MoE **+ vision** (Qwen3.8-Flash-Next) | `qwen-38-flash-next` (`flash-next`, `qwen38-next`) | 4090 + 5060 Ti + RAM (PLE on NVMe) | ~18.8 t/s · ⚠️ **~150 s cold swap-in — batch its work** |

All: 131072 context, MTP speculative decoding, thinking on (q8_0 KV — **except Flash-Next, which uses f16**;
q8_0 asserts on its QSA path). The 27B **and Flash-Next** are native **vision-language** models (images +
video). The 27B replaced the retired 3.6-27B dense line (Q4 + Q6); its legacy aliases (`qwen36`,
`qwen36-q6`, `qwen36-text`) carry forward so existing clients keep working.
The 27B has **two swappable quant profiles** — `UD-Q6_K_M` (imatrix, quality; `-ts 4,1`/MTP `n=5`) and
`plain Q6_K` (uniform, ~+8% faster) — toggled via `swap/config.yaml.q6km` / `.plain` (see USAGE.md).

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

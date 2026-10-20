# GPU identities for the run scripts. Pin by UUID, never by index: CUDA/driver/BIOS reordering would
# otherwise silently move a model to the other card with no error. Index order below is the order the
# run scripts assume (CUDA0 = 4090, CUDA1 = 5060 Ti), which -ts / -ot ...=CUDA1 / -mg rely on.
GPU_4090=GPU-7b440191-6dc3-6a7f-eeb8-2992fb60f8d0
GPU_5060TI=GPU-1e198012-434a-5248-4006-4269103c2c4c
GPU_BOTH="$GPU_4090,$GPU_5060TI"
# gpu_uuid 0|1|<uuid>: resolve a legacy index (e.g. DEV=0 in swap/config.yaml) to its UUID.
gpu_uuid() {
  case "$1" in
    0) echo "$GPU_4090" ;;
    1) echo "$GPU_5060TI" ;;
    GPU-*) echo "$1" ;;
    *) echo "gpu_uuid: unknown GPU '$1' (use 0, 1 or a GPU-... UUID)" >&2; return 1 ;;
  esac
}

#!/usr/bin/env bash
# One-line Colab runner (Colab terminal, after the notebook has mounted Google Drive):
#   curl -fsSL https://raw.githubusercontent.com/bachrathyd/SOSD.jl/gpu/gpu/colab_run.sh | bash
# or, with options:  ... | bash -s -- quick      (short benchmark)
#                    ... | bash -s -- testonly
#                    ... | bash -s -- setup      (only install; e.g. before the interactive charts)
# Installs Julia (juliaup) if missing, clones/updates the `gpu` branch, instantiates the
# gpu/ environment, then runs gpu/test_gpu.jl and gpu/bench_gpu.jl in the background.
# Everything is logged to <Drive>/Colab Notebooks/SOSD_GPU/runs/<date>_<gpu>/ (or to
# /content/sosd_runs/... when Drive is not mounted). Package output never reaches the
# terminal (progress spinners freeze the browser).
set -u
MODE="${1:-full}"
REPO=https://github.com/bachrathyd/SOSD.jl
BRANCH=gpu
DIR=/content/SOSD.jl
export PATH=/root/.juliaup/bin:$PATH

GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr ' ' '_')
BASE="/content/drive/MyDrive/Colab Notebooks/SOSD_GPU/runs"
[ -d "/content/drive/MyDrive" ] || BASE=/content/sosd_runs
OUT="$BASE/$(date +%Y%m%d_%H%M)_${GPU:-noGPU}"
mkdir -p "$OUT"
echo "GPU: ${GPU:-none}   cores: $(nproc)   results -> $OUT"

if ! command -v julia >/dev/null; then
    echo "installing Julia 1.12 (juliaup) ..."
    curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel 1.12 > "$OUT/juliaup.log" 2>&1
fi
if [ -d "$DIR/.git" ]; then
    (cd "$DIR" && git fetch -q --depth 1 origin $BRANCH && git reset -q --hard FETCH_HEAD)
else
    git clone -q -b $BRANCH --depth 1 $REPO "$DIR"
fi
(cd "$DIR" && git log -1 --format='code: %h %s')

run() {
    cd "$DIR"
    echo "[$(date +%T)] instantiate + precompile (log: setup.log)"
    julia --project=gpu -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' > "$OUT/setup.log" 2>&1
    if [ "$MODE" = "setup" ]; then echo "[$(date +%T)] done (setup only)"; return; fi
    echo "[$(date +%T)] GPU tests (log: test_gpu.log)"
    julia -t auto --project=gpu gpu/test_gpu.jl > "$OUT/test_gpu.log" 2>&1
    grep -a "Test Summary" -A2 "$OUT/test_gpu.log"
    if [ "$MODE" != "testonly" ]; then
        echo "[$(date +%T)] benchmark (log: bench_gpu.log)"
        BM=""; [ "$MODE" = "quick" ] && BM=quick
        julia -t auto --project=gpu gpu/bench_gpu.jl "$OUT" $BM > "$OUT/bench_gpu.log" 2>&1
    fi
    echo "[$(date +%T)] done"
}
nohup bash -c "$(declare -f run); DIR='$DIR'; OUT='$OUT'; MODE='$MODE'; run" > "$OUT/run.log" 2>&1 &
echo "running in the background (pid $!). Progress:"
echo "  tail -f \"$OUT/run.log\""
echo "  grep -a 'per ρ' \"$OUT/bench_gpu.log\""

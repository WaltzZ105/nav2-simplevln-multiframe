#!/usr/bin/env bash
set -euo pipefail

work_dir=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$work_dir/vendor${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export USE_HUB_KERNELS=NO TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

exec "${NAV2_PYTHON:-python}" -m torch.distributed.run --standalone \
    --nproc_per_node="${NAV2_NPROC:-4}" "$work_dir/train.py" "$@"

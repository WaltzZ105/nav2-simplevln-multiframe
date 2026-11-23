#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
upstream="${SIMPLEVLN_ROOT:?Set the official SimpleVLN checkout directory}"
expected=0216a1aa2d78e5f093595265d28bd90b29e0c87e
if [[ $(git -C "$upstream" rev-parse HEAD) != "$expected" ]]; then
    echo "Expected SimpleVLN checkout $expected" >&2
    exit 1
fi
root="${NAV2_CACHE_ROOT:?Set the cache output directory}"
protocol="${NAV2_PROTOCOL:?Set the rendered NAV2 protocol.json}"
model="${NAV2_MODEL:?Set the local Qwen3.5-4B directory}"
python="${NAV2_PYTHON:-python}"

"$python" "$here/cache_input.py" --protocol "$protocol" --output "$root/nav2.jsonl"
MAX_PIXELS=1003520 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$python" \
    "$upstream/sft/cache/fast_cache_dataset.py" --model "$model" --model_type qwen3_5 \
    --template qwen3_5 --enable_thinking false --dataset "$root/nav2.jsonl" \
    --dataset_num_proc "${NAV2_CACHE_WORKERS:-64}" --to_cached_dataset true \
    --output_dir "$root/stage1"

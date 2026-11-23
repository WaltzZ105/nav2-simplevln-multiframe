#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH="$here:$here/..:$here/../vendor${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAX_PIXELS=1003520
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NAV2_PROTOCOL="${NAV2_PROTOCOL:?Set the rendered NAV2 protocol.json}"
export NNODES="${NNODES:-4}" NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
export NODE_RANK="${NODE_RANK:?Set NODE_RANK for this machine}"
export MASTER_ADDR="${MASTER_ADDR:?Set MASTER_ADDR to the node-0 address}"
export MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${NAV2_SMOKE:-0}" == 1 ]]; then
    export NAV2_VERIFY_UPDATE=1
    schedule=(--train_iters 3 --save_steps 3 --micro_batch_size 1 --global_batch_size 2
              --max_length 6144 --dataloader_num_workers 0 --dataset_num_proc 1 --logging_steps 1)
else
    schedule=(--num_train_epochs 1 --save_steps 50 --micro_batch_size 2 --global_batch_size 256
              --max_length 40960 --dataloader_num_workers 16 --dataset_num_proc 32)
fi

exec "${NAV2_PYTHON:-python}" -m swift.cli._megatron.main sft \
    --model "${NAV2_MODEL:?Set the local Qwen3.5-4B directory}" \
    --model_type qwen3_5 --template qwen3_5 --enable_thinking false \
    --cached_dataset "${NAV2_CACHE:?Set the official exported Arrow train cache}" \
    --external_plugins "$here/plugin.py" \
    --tensor_model_parallel_size 2 --sequence_parallel true --packing true \
    --freeze_llm false --freeze_vit true --freeze_aligner true \
    --recompute_granularity full --recompute_method uniform --recompute_num_layers 1 \
    --finetune true --mtp_num_layers 0 \
    --cross_entropy_loss_fusion false --calculate_per_token_loss false \
    --lr 1e-5 --lr_warmup_fraction 0.05 --min_lr 1e-6 \
    --save_safetensors true --no_save_optim true --no_save_rng true \
    --output_dir "${NAV2_OUTPUT:?Set the Megatron output directory}" \
    "${schedule[@]}" "$@"

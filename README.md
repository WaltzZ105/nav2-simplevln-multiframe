# NAV2 × SimpleVLN 逐动作多帧训练

这套代码用已有 R2R/RxR 本地分片训练 NAV2 的 Qwen3.5-4B、原 LoRA、极坐标 heatmap 和三步动作头。先在 Habitat 回放已存动作前缀，从每个原子动作后的画面中调用 SimpleVLN 官方规则选历史，再用 ms-swift SFT 训练现有双头。启动器使用四卡 DDP。

## 1. 指定已有本地资产

| 路径 | 本地内容 |
| --- | --- |
| `DATA_ROOT` | 完整分片根目录，包含 `r2r/manifests/all_train.jsonl`、`rxr/manifests/all_train.jsonl` 及两套 `shards/`。使用 `train-and-dev` 时也需要各自的 `all_dev.jsonl`。 |
| `SCENES_DIR` | `mp3d/<scene>/<scene>.glb` 和对应 `.navmesh` |
| `R2R_EPISODES` | 原项目的 R2R train episode 标注 `train.json.gz`，用于补齐部分分片缺失的起始位姿 |
| `MODEL_PATH` | 本地基础 Qwen3.5-4B，包含权重、`config.json`、processor 和 tokenizer |
| `SIMPLEVLN_ROOT` | SimpleVLN 源码，固定提交 `0216a1aa2d78e5f093595265d28bd90b29e0c87e` |
| `SWIFT_ROOT` | ms-swift 源码，固定提交 `5ef0daa3d34d2c6a3f2ac705f0a58a67e0f13da2` |

回放环境按 SimpleVLN 上游要求配置 Habitat-Sim 与 Habitat-Lab 0.3.3，并安装 `DataProcess/requirements-convert.txt`。训练环境使用 Python 3.12、CUDA 版 PyTorch 2.6.0、上表的 ms-swift 和本仓库的 `training/requirements.txt`。两套环境可分开。在训练环境里安装依赖并让 Python 找到固定版本的 ms-swift。[SimpleVLN 回放环境说明](https://github.com/simplevln/SimpleVLN/tree/0216a1aa2d78e5f093595265d28bd90b29e0c87e/DataProcess#trajectory-collection)

~~~bash
export SWIFT_ROOT=/path/to/ms-swift
# 将 ms-swift 源码切到上表中的提交后再安装
python -m pip install -r training/requirements.txt -e "$SWIFT_ROOT"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NAV2_PYTHON="$(command -v python)"
~~~

四卡验收环境使用 `flash-linear-attention==0.3.2` 和 `causal-conv1d==1.5.0.post8`。可以按当前 Torch/CUDA 环境安装对应的加速依赖。先保证本机的 `import torch` 和 `torch.cuda.is_available()` 可用。

## 2. 从本地分片生成逐动作观测

在仓库根目录设置以下路径。观测缓存需要受训练节点共同访问，路径应保持一致。

~~~bash
export DATA_ROOT=/path/to/local/dataset
export SCENES_DIR=/path/to/scene_datasets
export R2R_EPISODES=/path/to/r2r/train/train.json.gz
export MODEL_PATH=/path/to/local/Qwen3.5-4B
export SIMPLEVLN_ROOT=/path/to/SimpleVLN
export OBS_ROOT=/path/to/primitive-observation-cache
export WORK_ROOT=/path/to/training-index

python tools/prepare_observations.py \
  --data-root "$DATA_ROOT" --upstream "$SIMPLEVLN_ROOT" --dataset r2r \
  --split train-and-dev --scenes-dir "$SCENES_DIR" --episodes "$R2R_EPISODES" \
  --output "$OBS_ROOT/r2r" --sim-gpu 0 --resume

python tools/prepare_observations.py \
  --data-root "$DATA_ROOT" --upstream "$SIMPLEVLN_ROOT" --dataset rxr \
  --split train-and-dev --scenes-dir "$SCENES_DIR" \
  --output "$OBS_ROOT/rxr" --sim-gpu 0 --resume
~~~

单档案回放时加 `--archive` 和本地清单里的相对 TAR 路径。每档案生成 `rgb/`、`observations.jsonl` 和 `receipt.json`。回放会核对已归档节点的位姿，按官方相机配置渲染逐动作 RGB，并调用官方采样函数选择最多 8 张历史帧，再加当前帧。原始决策点、heatmap 和三步动作标签来自同一条 NAV2 记录。渲染出的黑帧保留原像素，并在收据里计数。 [SimpleVLN 官方图像采样与提示](https://github.com/simplevln/SimpleVLN/blob/0216a1aa2d78e5f093595265d28bd90b29e0c87e/DataProcess/tool/convert_vln.py)

## 3. 建索引并训练

切换到训练环境，并选用一个尚未创建的索引输出目录。索引检查渲染收据、节点 ID 和原标签的 TAR 字节偏移。下述命令覆盖所有训练及内部开发样本，共 4,115,900 条，对应 128,622 次更新。每条样本曝光一次，顺序由固定种子决定。

~~~bash
python tools/build_train_protocol.py \
  --data-root "$DATA_ROOT" --observation-root "$OBS_ROOT" --model "$MODEL_PATH" \
  --output "$WORK_ROOT" --split train-and-dev

bash training/launch.sh --protocol "$WORK_ROOT/protocol.json" \
  --output /path/to/train-run --micro-batch 4 --workers 2 \
  --skip-recompute-layers 20 --save-steps 1000
~~~

默认四卡，全局 batch 32，每卡 4 条并累积 2 次。梯度重计算默认开启，学习率峰值为 `1e-5`，warmup 5%，余弦最低学习率为 `1e-6`。训练时有热图损失和有效格点动作损失。两路参数发生更新后，rank 0 会输出 `head_update_norms`。需要观察每步耗时和四卡峰值可加 `--measure-updates`。

恢复时沿用同一份索引、输出目录及训练参数，给启动命令追加 `--resume /path/to/train-run/checkpoint-N`。目录内 `nav2.pt` 保存 LoRA、投影和双头，另外保存优化器、调度器和各卡 RNG。

### 小规模验收

先用 `--archive` 渲染一个或几个 R2R 档案，再把 `$OBS_ROOT/r2r` 作为验收索引的观测根目录。验收模式会重复取样，生成完整的四卡 batch。

~~~bash
export SMOKE_INDEX=/path/to/smoke-index
export SMOKE_RUN=/path/to/smoke-run

python tools/build_train_protocol.py \
  --observation-root "$OBS_ROOT/r2r" --model "$MODEL_PATH" \
  --output "$SMOKE_INDEX" --smoke-updates 8

COMMON=(--protocol "$SMOKE_INDEX/protocol.json" --output "$SMOKE_RUN" --steps 128622
  --workers 2 --micro-batch 4 --skip-recompute-layers 20 --measure-updates)
bash training/launch.sh "${COMMON[@]}" --stop-after 4
bash training/launch.sh "${COMMON[@]}" --stop-after 8 --resume "$SMOKE_RUN/checkpoint-4"
~~~

验收沿用全量的学习率调度长度，实际执行 8 次更新。日志应显示两路损失、`head_update_norms` 和 `completed_update` 到达 8。

已用 R2R 和 RxR 的 60 条真实样本在 4 张 A800 上完成 8 次更新，第 4 次存档，第 5 至 8 次从断点恢复。真实观测回放在 Habitat 0.2.4 兼容环境做了小样测试。全量回放、0.3.3 环境和全量训练待验收。

本地数据逻辑测试

~~~bash
SIMPLEVLN_ROOT="$SIMPLEVLN_ROOT" python -m unittest discover -s tests -v
~~~

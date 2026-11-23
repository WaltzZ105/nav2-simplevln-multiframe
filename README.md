# NAV2 × SimpleVLN 多帧交接

该版本把 SimpleVLN 官方的逐动作多帧处理接入此前的四卡 ms-swift 训练入口，保留 NAV2 的 Qwen3.5-4B、原 LoRA、输入投影、heatmap 和三步动作头。 [NAV2 模型与双头](https://github.com/WaltzZ105/nav2/tree/be608fed09738df5333b41e4982f6f05571043be/vendor)

[SimpleVLN 官方 Megatron 训练入口与验收](training/megatron/README.md) 使用下文相同的数据准备，并附有独立的环境和启动命令。

## 数据从哪里来

| 输入 | 来源 | 用途 |
| --- | --- | --- |
| 约 235 GB 的状态分片和清单 | [Hugging Face 固定版本](https://huggingface.co/datasets/WaltzZ105/multiroute-state-shards/tree/218857bc3632d6895e2b16560af4cd03d74236e4) | 指令、动作前缀、位姿和双头 NPZ 标签 |
| MP3D 场景及配套 navmesh | [Matterport3D](https://github.com/niessner/Matterport) | Habitat 逐步执行并渲染 RGB |
| 原项目使用的 R2R train episode 标注 | [VLN-CE](https://jacobkrantz.github.io/vlnce/data) | 补齐 15 个分片缺失的起始位姿 |
| SimpleVLN 固定源码 | [SimpleVLN DataProcess](https://github.com/simplevln/SimpleVLN/tree/0216a1aa2d78e5f093595265d28bd90b29e0c87e/DataProcess) | 官方历史采样、首帧处理、提示和相机配置 |

HF 有 16,536 个 TAR 和 4,115,900 条状态记录。四份清单的 SHA-256 与原训练机逐一匹配。实际有样本的档案为 16,299 个。其中 15 个 R2R train 档案缺 step 0 位姿，原项目的 R2R train 标注包含对应 15 个 episode。已发布的图像随三步决策状态存储。完整多帧要从根姿态重放基本动作。构建脚本直接引用 HF TAR 中原有的 NPZ 偏移，保留 heatmap 及三步动作监督。具备上述资产的接手人可以在本地构建 RGB 缓存，原 HF 数据可直接复用。

## 1. 下载和审计

在仓库根目录创建数据工作区。渲染需按官方说明在 Linux 上安装 Habitat-Sim 与 Habitat-Lab 0.3.3。 [SimpleVLN 环境说明](https://github.com/simplevln/SimpleVLN/tree/0216a1aa2d78e5f093595265d28bd90b29e0c87e/DataProcess#trajectory-collection)

~~~bash
git clone https://github.com/simplevln/SimpleVLN.git third_party/SimpleVLN
git -C third_party/SimpleVLN checkout --detach 0216a1aa2d78e5f093595265d28bd90b29e0c87e
python -m pip install huggingface_hub -r third_party/SimpleVLN/DataProcess/requirements-convert.txt

export DATA_ROOT=/path/to/hf-snapshot
python tools/fetch_data.py --root "$DATA_ROOT" --manifests-only
python tools/prepare_observations.py \
  --data-root "$DATA_ROOT" --upstream third_party/SimpleVLN \
  --dataset r2r --split train-and-dev --audit \
  --episodes /path/to/original-r2r/train/train.json.gz
python tools/prepare_observations.py \
  --data-root "$DATA_ROOT" --upstream third_party/SimpleVLN \
  --dataset rxr --split train-and-dev --audit
~~~

只下载清单约需 7.7 GB。脚本在固定 HF 版本下载并验证文件大小和 SHA-256。审计输出包含状态数、档案数、缺失的历史帧引用数和待补起点列表。正式构建前确认两个审计输出里的 `missing_roots` 都为空。

已经同步完整快照时可加 `--local-only` 校验现有文件。只有 HTTPS 下载镜像可达时，可加 `--resolve-endpoint https://hf-mirror.com`，校验固定版本的校验表和下载内容。

## 2. 回放并生成观测

先选一个样本档案检查 Linux 渲染环境。例子来自发布集的 R2R train。

~~~bash
export SCENES_DIR=/path/to/scene_datasets
export OBS_ROOT=/path/to/observation-cache
export SHARD=r2r/shards/000/ep10000_b57f13f46b.tar

python tools/fetch_data.py --root "$DATA_ROOT" --archive "$SHARD"
python tools/prepare_observations.py \
  --data-root "$DATA_ROOT" --upstream third_party/SimpleVLN \
  --dataset r2r --split train --archive "$SHARD" \
  --scenes-dir "$SCENES_DIR" --output "$OBS_ROOT/r2r" --sim-gpu 0
~~~

`SCENES_DIR` 下应有 `mp3d/<scene>/<scene>.glb` 和对应的 `.navmesh`。全量构建用 `fetch_data.py --all`，分别对 R2R 和 RxR 运行，去掉 `--archive`。复用此前全量训练的样本范围时用 `--split train-and-dev`。R2R 要加原项目 train 标注的 `--episodes`。中断后用相同输入加 `--resume`，脚本核对已完成档案的收据和图像路径。

回放从同档案的真实起点出发，沿动作前缀渲染每一步 RGB。两条分支共享动作前缀时复用图像。遇到已归档的决策点，脚本比较位置和旋转，发现偏差即停止。当前帧与历史帧均用官方相机配置渲染，统一为 640×480、90° HFOV。脚本在逐动作帧序列上直接调用官方采样和提示函数，均匀抽取至多 8 张历史图，再加当前图。首个决策点依官方函数重复首帧。沿用官方采集对真实模拟器输出的处理方式，渲染出的黑帧会随原像素保留，并在收据里计数。

原档案的决策点继续定义监督样本。逐动作观测补齐的是输入序列。上游 `convert_episode_vln` 的合并动作规则为动作文本重新划分决策点，接入时调用的是它的官方图像采样与提示函数。

每档案生成 `rgb/`、`observations.jsonl` 和 `receipt.json`。观测 JSONL 记录官方提示、图像路径、精确帧步数和 NAV2 标签引用。生成记录使用绝对路径，集群节点需要共享相同挂载路径。单盘要同时容纳 HF 快照和逐动作缓存，缓存体积以实际生成结果为准。

已在既有 Habitat 0.2.4 CPU 环境对三个真实档案各取 12 条状态做回放，分别生成 34、40、34 张逐动作帧，三个试例均通过归档位姿校验。其中缺 step 0 的 R2R 试例用原始 episode 标注恢复起点，在 40 张帧里记录到 6 张黑帧。0.2.4 冒烟使用了对应的临时配置兼容处理。官方 0.3.3 环境及全量回放待验收。

## 3. 四卡训练环境

训练环境使用 Linux、Python 3.12、与 NVIDIA 驱动兼容的 CUDA PyTorch 2.6.0，以及固定版本的 ms-swift。可以把渲染和训练放在不同环境。准备本地 Qwen3.5-4B 权重目录，训练入口使用离线模型路径。 [Qwen3.5-4B 基础模型](https://huggingface.co/Qwen/Qwen3.5-4B)

~~~bash
git clone https://github.com/modelscope/ms-swift.git third_party/ms-swift
git -C third_party/ms-swift checkout --detach 5ef0daa3d34d2c6a3f2ac705f0a58a67e0f13da2
python -m pip install -r training/requirements.txt -e third_party/ms-swift

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NAV2_PYTHON="$(command -v python)"
export MODEL=/path/to/Qwen3.5-4B
~~~

A800 实测环境使用 `flash-linear-attention==0.3.2`、`fla-core==0.3.2` 和 `causal-conv1d==1.5.0.post8` 的 CUDA 兼容构建。复现实测显存与吞吐时也应使用兼容的内核构建。

训练入口使用 ms-swift 的 `Seq2SeqTrainer` 和 DDP，双头直接读取原热图和动作监督。上游公开的 `megatron sft` 启动器对应 SimpleVLN 的动作文本训练配方。 [SimpleVLN Stage 1 官方启动器](https://github.com/simplevln/SimpleVLN/blob/0216a1aa2d78e5f093595265d28bd90b29e0c87e/sft/stage1/train_qwen_35_4b_stage1.sh)

## 4. 小样开训并恢复

先用第 2 节构建的 R2R 小样。索引会重复曝光小样，凑够 8 次更新，用于验证读取、训练、存档和恢复。

~~~bash
python tools/build_train_protocol.py \
  --observation-root "$OBS_ROOT/r2r" --model "$MODEL" \
  --output work/pilot --smoke-updates 8

COMMON=(--protocol work/pilot/protocol.json --output runs/pilot --steps 128622 \
  --workers 2 --micro-batch 4 --skip-recompute-layers 20 --measure-updates)
bash training/launch.sh "${COMMON[@]}" --stop-after 4
bash training/launch.sh "${COMMON[@]}" --stop-after 8 --resume runs/pilot/checkpoint-4
~~~

验收日志应包含两路损失、两个头加上 LoRA 和输入投影的非零权重更新，以及 `completed_update` 到达 8。目录下会有 `checkpoint-4` 和 `checkpoint-8`。先使用空输出目录执行第一条启动命令。

### A800 实测

在 4 张 80 GiB A800 上使用 60 条真实 R2R/RxR 状态，输入覆盖 2–9 张观测，其中 53 条使用 9 张。三份标签 TAR 的 SHA-256 与 HF 发布版本逐一匹配。用每卡 micro-batch 4、累积 2 次和此前的部分梯度重计算配置，从头训练到第 4 步、存档、退出，再加载断点完成第 8 步。两路损失均为有限值，两个头、LoRA 和投影在启动和恢复后均发生权重更新。

每卡板载显存实测峰值分别为 63.2、62.2、62.2、62.1 GiB，约占 78–79%。排除首次更新和恢复后首次更新，约 5.4 秒/步。这次验证覆盖已有训练环境中的短样本训练和断点恢复。新机器从零安装及全量训练用时待验证。

## 5. 全量训练

R2R 与 RxR 的图像缓存分别放在 `$OBS_ROOT/r2r`、`$OBS_ROOT/rxr`。下面使用此前的全量范围，把 183,176 条内部开发样本也用于训练。选择 `--split train` 可只用发布集的 train 清单。

~~~bash
python tools/build_train_protocol.py \
  --observation-root "$OBS_ROOT" --data-root "$DATA_ROOT" \
  --split train-and-dev --model "$MODEL" --output work/full
bash training/launch.sh --protocol work/full/protocol.json \
  --output runs/full --workers 2 --micro-batch 4 \
  --skip-recompute-layers 20 --save-steps 1000
~~~

全量索引按原训练清单顺序核对样本、决策步和 NPZ 偏移，再以原种子生成一次曝光顺序。全局 batch 为 32。恢复时沿用原索引、启动参数和输出目录，加 `--resume runs/full/checkpoint-步数`。保留已经生成的路径和索引文件，以便从相同数据位置继续。

### 本地数据逻辑检查

~~~bash
SIMPLEVLN_ROOT="$PWD/third_party/SimpleVLN" python -m unittest discover -s tests -v
~~~

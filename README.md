# NAV2 × SimpleVLN 多帧交接

**当前交付是数据下载、审计和逐动作观测构建。官方 Megatron SFT 双头接入及 GPU 验收仍待完成。**

目标是保留 NAV2 的 Qwen3.5-4B、原 LoRA、极坐标 heatmap 和三步动作头，使用 SimpleVLN 官方的多帧处理和 R2R/RxR SFT Stage 1。 [NAV2 模型与双头](https://github.com/WaltzZ105/nav2/tree/be608fed09738df5333b41e4982f6f05571043be/vendor)

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

## 3. SFT 接入与验收

官方 Qwen3.5-4B Stage 1 启动器采用 `megatron sft`、打包训练、张量并行 2 和 4 节点 × 8 GPU。它以动作文本做 token 交叉熵。NAV2 的监督来自一个决策点对应的 heatmap、有效位置掩码及 40 类软动作序列分布。保留 NAV2 输出并使用官方训练引擎，需要在 Megatron 每条打包序列上取出隐藏状态，挂接原投影与双头，并把原损失接进训练和保存流程。 [SimpleVLN Stage 1 官方启动器](https://github.com/simplevln/SimpleVLN/blob/0216a1aa2d78e5f093595265d28bd90b29e0c87e/sft/stage1/train_qwen_35_4b_stage1.sh)

LoRA 也需要逐层验收。原模型在全注意力层分别给 Q、V 两个投影挂独立 LoRA，在门控线性注意力层只给 `in_proj_qkv` 挂 LoRA。官方 Megatron 桥会合并投影，直接套用融合 QKV LoRA 会改变参数化。正式训练交接以原 LoRA 覆盖一致、官方打包图像与各自标签一一对应、双头和 LoRA 都完成更新、权重能保存并重新加载为验收条件。

### 本地数据逻辑检查

~~~bash
SIMPLEVLN_ROOT="$PWD/third_party/SimpleVLN" python -m unittest discover -s tests -v
~~~

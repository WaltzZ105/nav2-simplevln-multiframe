# SimpleVLN Stage 1 与 NAV2 双头

此入口运行 SimpleVLN 发布配方所用的 ms-swift `megatron sft`，复用官方缓存导出、packing、张量并行和 optimizer loop。默认沿用官方 4 节点 × 8 GPU、TP 2、全局 batch 256、1 epoch 和学习率计划。 [SimpleVLN Qwen3.5-4B Stage 1 配方](https://github.com/simplevln/SimpleVLN/blob/0216a1aa2d78e5f093595265d28bd90b29e0c87e/sft/stage1/train_qwen_35_4b_stage1.sh)

NAV2 适配在原有决策点读取逐动作观测。官方 tokenizer 和 collator 打包后，根据序列边界将隐藏状态交给原投影和双头。梯度目标保留热图和三步动作损失。Q 和 V 的 LoRA 参数独立，GDN 层只在原 QKV 投影所对应的通道上加 LoRA。冻结的 Qwen 权重、LoRA rank 和双头尺寸沿用 NAV2 版本。脚本关闭词表交叉熵融合及额外的 MTP 目标。

## 准备数据与训练环境

按主 README 生成逐动作帧与 `protocol.json`。所有节点需要用相同绝对路径访问场景回放输出、HF 标签和模型。使用 SimpleVLN 固定提交 `0216a1aa2d78e5f093595265d28bd90b29e0c87e` 和 ms-swift 固定提交 `5ef0daa3d34d2c6a3f2ac705f0a58a67e0f13da2`。训练环境需要一致的 PyTorch、CUDA、Transformer Engine、Megatron-Core 和 mcore-bridge。 [Megatron-SWIFT 安装说明](https://swift.readthedocs.io/en/v4.1/BestPractices/Qwen3_5-Best-Practice.html)

~~~bash
export SIMPLEVLN_ROOT=/path/to/SimpleVLN
export NAV2_MODEL=/path/to/local/Qwen3.5-4B
export NAV2_PROTOCOL=/path/to/prepared/protocol.json
export NAV2_CACHE_ROOT=/path/to/new/cache
export NAV2_PYTHON="$(command -v python)"

bash training/megatron/cache.sh
export NAV2_CACHE="$NAV2_CACHE_ROOT/stage1/train"
~~~

每个缓存样本保留来自 `protocol.json` 的标签引用。原始张量从 HF TAR 中按字节偏移读取。训练器保存官方格式的 Stage 1 权重和独立的 `nav2.pt` 双头、投影、LoRA 权重。

## 启动

在每个节点设置 `NODE_RANK`，并使用相同的 `MASTER_ADDR` 和 `MASTER_PORT`。生产参数按官方 32 卡脚本配置。

~~~bash
export NODE_RANK=0
export MASTER_ADDR=192.0.2.10
export MASTER_PORT=29500
export NAV2_OUTPUT=/path/to/new/megatron-run
bash training/megatron/launch.sh
~~~

单机四卡冒烟保留 TP 2 和 packing，使用小 batch 执行 3 次更新。使用少量已渲染样本构建验收协议和缓存后，在空目录启动。

~~~bash
export NNODES=1 NPROC_PER_NODE=4
export NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NAV2_SMOKE=1 NAV2_OUTPUT=/path/to/new/megatron-smoke
bash training/megatron/launch.sh --attention_backend auto --check_model false
~~~

验收检查两路损失、`head_and_adapter_updates`，以及第 3 步导出的 `nav2.pt`。官方 Stage 1 配置按 weights-only 保存。需要接着 optimizer 和 RNG 状态续训时，应另行验证训练状态的保存与恢复。

## A800 验收

用 HF 分片中的 60 条真实 R2R/RxR 样本通过官方缓存导出，再在 4 张 A800 上以 TP 2、DP 2、packing 开启完成 3 次 optimizer 更新。两路损失有限，heatmap 头、动作头、LoRA 和投影均确认更新。第 3 步成功导出 safetensors 和 NAV2 权重。

验收环境为 PyTorch 2.11.0+cu130、CUDA 13.0、Transformer Engine 2.21.0、Megatron-Core 0.18.0 和 mcore-bridge 1.7.0。A800 的打包 BF16 短测使用 `--attention_backend auto`，由 Transformer Engine 选择 unfused 内核。默认入口沿用官方的 FlashAttention 设置。32 卡和全量训练待实测。

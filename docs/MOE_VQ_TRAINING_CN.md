# MOMENT + MoE + 输入 VQ 单卡训练

入口为 `scripts/pretraining/train_moe_vq.py`，Python 包名为 `moment`。输入经过 patch embedding，先做 VQ 编码、最近邻量化、解码回 `d_model`，再进入带稀疏 MoE FFN 的 T5 encoder。训练目标是隐藏位置的重建 MSE，加 `0.01 × MoE 平衡损失 + 1.0 × VQ 损失`。padding 和隐藏占位 token 不参与 VQ 拟合；最终记忆库只收集最终权重对训练集未遮挡输入激活的代码。

## 1. 服务器环境

建议新建 Python 3.10/3.11 环境，先安装与显卡驱动匹配的 PyTorch，再安装最小依赖。以下版本组合为可复现的服务器安装建议；本地实际执行环境与验证记录应以交付说明为准，不把建议版本当作已在服务器验证。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 示例：CUDA 12.4 wheel；请按服务器驱动选择对应官方 wheel。
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-cloud.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

CPU 环境可将 wheel 地址换成 `https://download.pytorch.org/whl/cpu`。此入口不依赖 W&B、PILE、原项目 `.env` 中的数据路径，不需要 `torchvision`、`datasets` 或模型下载。请勿用原始 `requirements.txt` 覆盖这个专用环境；`python-dotenv` 为可选的 `.env` 加载支持。

另一份 `requirements-verified.txt` 记录本地实际执行的核心版本：Python 3.13.5、PyTorch 2.11.0+cu128、Transformers 5.4.0、huggingface-hub 1.8.0、NumPy 2.3.1、SciPy 1.16.0、PyYAML 6.0.2。若使用它，先安装适合服务器的 `torch==2.11.0` wheel，再安装此文件；这份版本记录不等于已在目标 Linux 服务器执行过安装。服务器 GPU 若需要更新的 CUDA/计算能力支持，应选相应 PyTorch wheel。

上述 wheel 示例可对照 [PyTorch 官方历史版本安装页](https://pytorch.org/get-started/previous-versions/)，Transformers 4.44.2 的依赖约束可对照其 [官方发布源码](https://github.com/huggingface/transformers/blob/v4.44.2/setup.py)。

本次本地实际验证使用 Windows、Python 3.13.5、PyTorch 2.11.0+cu128、Transformers 5.4.0、NumPy 2.3.1、PyYAML 6.0.2：最终 48 个 unittest 全部通过；CPU 离线 smoke 跑至 3 步再续到 5 步，与一次直跑 5 步的模型、AdamW 状态和 Python/NumPy/Torch 随机状态完全一致。最终 bank 的 192 次选码来自 24 个训练窗口 × 2 通道 × 4 个 patch。RTX 5080 上另验证了 FP16 溢出自动回退后完成 3 次成功更新，以及默认 256 维/4 层/8 专家配置的 2 步 CUDA FP16 训练与完整 bank 构建。该结果说明入口流程可执行，不代表推荐服务器环境已逐一安装验证，也不代表真实数据训练效果。

## 2. 先跑离线小模型

从项目根目录运行：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/pretraining/train_moe_vq.py --smoke --device cpu --output-dir runs/smoke
python scripts/pretraining/train_moe_vq.py --smoke --device cuda:0 --output-dir runs/smoke-cuda
```

`--smoke` 使用确定性合成数据和完全离线的随机 T5：`d_model=32, d_ff=64, num_layers=1, num_heads=2, d_kv=16`，序列长度 32、patch 长度 8、遮挡率 0.5，默认 2 个优化器更新。它验证训练、检查点和记忆库生成流程，不代表收敛或下游精度。CPU 自动关闭 AMP；CUDA 默认配置支持 fp16 AMP，smoke 为方便核查使用 fp32。

```bash
# --max-steps 是从初始训练开始累计的总步数，不是额外训练步数。
python scripts/pretraining/train_moe_vq.py --resume runs/smoke/checkpoint_final.pt --max-steps 4 --device cpu --output-dir runs/resumed
```

## 3. 真实数据格式

推荐连续数据 `[T,C]`。CSV 第一行必须是唯一列名，后面按时间升序排列；全部为非数值的列（如 ISO 日期字符串）会被排除，完整数值列作为通道。数值列中的空值、混合文本、NaN/Inf 会报错；数字形式的时间戳也会被当作通道，请在导出时删除。NPY 必须是实数数组，禁止 pickle/object 数组。

```text
date,temperature,pressure
2024-01-01 00:00:00,21.5,101.2
2024-01-01 00:01:00,21.7,101.1
...
```

在配置 `data.layout: TC` 时，CSV/NPY 都按 `[T,C]` 解析。先按原始时间轴切分训练/验证/测试（默认 80%/10%/10%），只在训练段拟合通道均值和标准差，再各自滑窗，绝不跨切分边界。`seq_len=512, stride=512` 表示不重叠训练窗口；每段末尾不足一个完整窗口的部分不使用，不自动填补。每段至少需要一个完整窗口；例如默认比例建议至少 5120 行，边界取整可能要求更多。先检查数据是否等间隔、按时间排序。

若 NPY 已是 `[N,C,L]`，显式设置 `data.layout: NCL`，且 `L == model.seq_len`。程序按 N 的顺序切分；NPY 没有时间戳可验证历史重叠，必须由数据提供方保证输入窗口按时间排序且跨切分边界不共享原始时间点。对于现有重叠滑窗，应重新提供连续 TC 数据以保证时间隔离，不可直接随机拆分窗口。

```bash
python scripts/pretraining/train_moe_vq.py --config configs/pretraining/moe_vq.yaml --data data/train.csv --output-dir runs/experiment --device cuda:0 --max-steps 10000
# 使用同一主配置和较大的合成数据，不是 --smoke 小模型。
python scripts/pretraining/train_moe_vq.py --synthetic --output-dir runs/synthetic --device cuda:0 --max-steps 20
```

主配置也是离线随机初始化的 T5 encoder（256 维、4 层），不是下载的预训练 MOMENT 参数。`transformer_config` 决定离线骨干结构；增大结构时保证 `model.d_model` 与其一致。若需要 Hugging Face 预训练 T5，删除 `transformer_config`、将 `randomly_initialize_backbone` 设为 false，设置受支持的 `transformer_backbone` 和匹配的 `d_model`；这需要网络或已缓存权重，且来源是 T5 预训练骨干。8 个专家是原 FFN 的拷贝，不会把全部模型参数都乘以 8；FFN 参数和优化器状态会增加，即便 top-k=2 只派发到被选专家。支持 top-k=2/3/4，且不能超过专家总数。首版单卡训练关闭梯度检查点。

## 4. 输出和恢复

每次运行生成：

| 文件 | 内容 |
|---|---|
| `config.resolved.yaml` | 实际使用的模型/数据/优化配置 |
| `metrics.jsonl` | 每步总损失、重建损失、辅助损失、梯度范数、验证损失、完成记录 |
| `train_transforms.npz` | 只由训练段拟合的通道缩放参数 |
| `checkpoint_latest.pt` | 定期检查点；正常结束后含最终记忆库 |
| `checkpoint_final.pt` | 最终模型、优化器、scaler、累计 step、下一批游标、随机状态、配置和数据摘要 |
| `codebook_memory.pt` | 完整 `codebook`、最终训练遍历的 `memory_counts`、`memory_indices`、直接可用的 `memory_bank`、历史训练诊断 `activation_counts` 及配置 |

默认每 100 步保存检查点；退出中断后从最近一次已完整保存的检查点继续。临时文件通过替换写入，降低半写文件风险。恢复直接使用检查点中的配置；`--config` 和 `--smoke` 不会覆盖它，允许 `--max-steps` 改累计目标、`--device` 改设备、`--output-dir` 改输出位置。原数据仍须可用，也可通过 `--data` 指定相同内容的新路径；内容和形状摘要必须一致。

恢复保存 Python/NumPy/PyTorch/CUDA RNG、AdamW/scaler 状态、epoch 和下一 batch 游标。每个 epoch 的窗口顺序由 seed+epoch 独立生成，固定 `num_workers=0`，包括最后一个不满 batch；跳过已消费 batch 不会再次抽取遮挡。验证运行前后恢复 RNG，不改变后续训练遮挡。相同软件/设备设置下可重复续训；跨 CPU/GPU、AMP 或库版本的浮点结果不能保证逐位一致。累计目标已达到时不会额外执行优化器更新，仍会重建最终记忆库。

检查点加载使用 `torch.load(..., weights_only=False)`，因为其中包含优化器及随机状态，只加载可信来源文件。这个入口不会创建多卡进程或自动分布式训练。

## 5. 单独重建/导出记忆库

```bash
python scripts/pretraining/train_moe_vq.py --resume runs/experiment/checkpoint_final.pt --build-memory-only --device cuda:0 --output-dir runs/rebuilt-bank
```

这条命令不执行优化器更新，使用保存的训练切分和缩放参数，以 eval 模式、无人工遮挡遍历全部训练窗口，重置旧激活统计，再导出新 `codebook_memory.pt` 和带 bank 的最终检查点。验证、测试和训练过程的历史命中都不加入最终 bank。不会丢弃最后一个不满 batch；`max_validation_batches` 仅限制验证，与最终 bank 遍历无关。重叠训练窗口会重复统计其 token 出现次数。

完整可学习码本默认 `256 × 64` float32，约 64 KiB；被最终训练遍历激活的 M 个代码构成记忆库，`M ≤ 256`。导出文件包含 `codebook`、`memory_bank`、`memory_indices`、`memory_counts`、诊断用 `activation_counts` 和 VQ 配置。MoE 路由 top-k 和 VQ 最近邻选码是两个不同过程。完整模型/优化器检查点通常远大于码本文件。

FP16 的 GradScaler 如果检测到溢出，会降低缩放值、跳过当前更新，并记录 `amp_overflow`、`grad_norm: null`；该批不计入成功优化步数。连续溢出默认上限 100，可在 `training.max_consecutive_overflows` 调整；达到上限先保存最新检查点再报错。未启用 scaler 时出现非有限梯度则直接报错。

消融实验可在配置中关闭 `use_moe` 或 `use_vq`。关闭 VQ 的正常训练会保存模型、`bank_ready=False`，不导出记忆库；此时 `--build-memory-only` 会在训练和创建输出目录前拒绝执行。

恢复模型和单独加载 bank 的示例：

```python
import torch
from moment.models.moment import MOMENT

checkpoint = torch.load("runs/experiment/checkpoint_final.pt", map_location="cpu", weights_only=False)
config = dict(checkpoint["config"]["model"])
config["device"] = "cpu"
model = MOMENT(configs=config)
model.load_state_dict(checkpoint["model"])
model.vq_vae.load_memory("runs/experiment/codebook_memory.pt")
model.eval()
# 应用相同 train_transforms.npz，输入 [batch, channels, seq_len]。
# 由 checkpoint 恢复时 persistent bank buffers 已在 model state_dict 中。
print(model.vq_vae.memory_indices, model.vq_vae.memory_bank.shape)
```

不要把验证/测试数据传给 `begin_memory_collection()` 后的模型。下游复用应加载完整模型权重及兼容的 persistent bank buffers，仅导入码本文件不会恢复整个预测模型。

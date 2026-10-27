# 改动与验证说明

交付日期：2026-10-24。

## 目录状态

改动写入 `D:/pycharm2025/DHI/moment-research-main/baseline`。原 `D:/pycharm2025/DHI/moment-research-main/moment-research-main` 未修改；实施前两份目录均为 186 个文件且逐文件 SHA-256 一致。同步后另用修改前哈希清单复核原项目。

## 主要改动

- `moment/models/layers/moe_block.py`：复制原 T5 FFN 为专家，真实稀疏 top-k=2/3/4 路由，padding 排除、负载均衡辅助损失。
- `moment/models/layers/vq_vae.py`：简单线性 VQ encoder、向量量化、decoder 投影；隐藏 token 原样绕过。完整码本、最终激活项的 bank 向量、索引及计数可单独导出/加载，也随 state_dict 保存。
- `moment/models/moment.py`：按用户选择，VQ 紧接 patch embedding、在 T5 之前；各条常用前向路径共享这一结构。支持离线 T5 配置、增强开关、显式辅助损失、旧 dense FFN 权重到专家的迁移。
- `moment/tasks/`：原预训练/微调接入辅助损失，修正停止条件与外部第二张 GPU 假设，训练后遍历训练集构建 bank；下游迁移保留 buffer，最新 checkpoint 查找忽略 memory 文件。
- `scripts/pretraining/train_moe_vq.py`：轻量单 CPU/GPU 入口，CSV/NPY 数据、按时间先切分后滑窗、训练段缩放、AMP、JSON 日志、完整断点恢复与独立 bank 重建。支持关闭 MoE/VQ 的消融配置。
- `configs/pretraining/moe_vq.yaml`、两份专用依赖文件、中文训练说明及 6 个测试文件。

当前版本包名是 `moment`，没有另建一个与现有代码脱节的 `momentfm` 目录。

## 验证结果

实际环境：Windows，Python 3.13.5，PyTorch 2.11.0+cu128，Transformers 5.4.0，RTX 5080。

- `python -m unittest discover -s tests -p "test_*.py" -v`：48 项全部通过，33.241 秒。
- MoE：top-k=2/3/4、只执行选定专家、梯度、padding、全无效输入和非法配置。
- VQ：nearest-code 与手算损失、straight-through 梯度、隐藏/无效输入绕过、最终 bank 排除历史及评估命中、完整导出校验。
- MOMENT：本地 tiny T5 的前向/反向、非默认 patch 长度、重建/embedding/forecast 路径、checkpoint 恢复、dense 到 MoE 权重迁移。
- CPU：3 步后恢复到累计 5 步，与直接 5 步的模型张量、AdamW 状态及 Python/NumPy/Torch RNG 完全一致。独立 bank 重建不改变权重/步数。
- 最终训练集 bank：24 个训练窗口 × 2 通道 × 4 patch = 192 次激活，验证/测试各 2 个窗口不参与。
- CUDA FP16 / top-k=3：初始 scale 65536 溢出后自动降至 32768，跳过该批且不计步，随后完成 3 次成功更新。另验证连续溢出上限会保存 checkpoint 再停止。
- 默认配置 CUDA FP16：256 维、4 层、8 专家、top-k=2，完成 2 次成功更新；完整遍历 12 个训练窗口，激活 31 个码本项。
- 兼容回归：无辅助字段的旧模型训练、关闭 VQ 的消融、最新 checkpoint 目录含 bank 文件均有测试。
- 独立审查与修正后复审：未留待处理的重要问题。

## 使用边界

此实现借鉴 COMET 的 VQ 与激活码本 bank；按用户选择采用输入侧量化，并非 COMET 全部方法的复现。主云配置为较小随机初始化 T5，适合验证和训练入口，不包含预训练 MOMENT checkpoint 或真实数据。新增层应重新训练。

本次未在实际 Linux 云服务器安装依赖或启动真实数据的正式长训练。服务器建议依赖版本与本地实测版本分别记录，不能把本机数步运行理解为收敛或下游精度提升。原 PILE/W&B 研究入口完整流程未做端到端数据训练，推荐先使用中文说明中的轻量入口。

代码包仅包含源码、配置、测试和说明，不包含本机测试 checkpoint、合成数据或临时文件。


## 2026-10-27 上传修复

GitHub push protection 检出了原始项目 llmtime.py 中继承的两处硬编码 Hugging Face token。现已移除字面量，改为读取可选环境变量 HF_TOKEN；未设置或为空时允许使用 Hugging Face 缓存登录。真实 token 不写入代码或示例文件。

新增 3 项认证回归测试，覆盖设置、未设置、空环境变量；全部通过。修复同步到 baseline 和此交付代码包。原始 moment-research-main 目录保持原样，不包含在上传目录中。对将推送的 Git 可达提交和打包文件进行常见凭据格式扫描，不通过绕过保护的方式上传。

如果旧 token 属于你的账号，请在 Hugging Face 的 Access Tokens 设置中撤销并重新创建。此修复没有使用或验证该凭据。

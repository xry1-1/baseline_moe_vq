# MOMENT MoE + 输入 VQ-VAE（用户确认版）

用户于 2026-10-24 确认继续实施，并明确两个变更：VQ 紧接 patch embedding、放在 T5 之前；所有代码改动放在 `baseline`，原 `moment-research-main` 保持修改前内容。

## 目录与结构

- 开发目标：`D:/pycharm2025/DHI/moment-research-main/baseline`。
- 原始项目：`D:/pycharm2025/DHI/moment-research-main/moment-research-main`。实施前确认 186 个文件与备份完全一致，无需撤销代码改动。
- 此研究版包名为 `moment`，新增文件放在 `moment/models/layers/`。

```text
时序 -> RevIN -> patch embedding -> VQ encoder -> codebook量化
     -> VQ decoder投影 -> T5 encoder（FFN替换为MoE） -> 原预测/重建头
```

## 已确认的功能

1. 每层 T5 的 FFN 替换为稀疏 MoE，保留注意力、归一化和残差。top-k 支持 2、3、4，默认 2；默认 8 个专家。只执行选中的专家，提供路由均衡损失和使用统计。
2. 采用简单 VQ-VAE 瓶颈，默认 latent 维度 64、码本 256 项；输入维度保持不变。隐藏占位 token 和 padding 原样绕过 VQ，不参与拟合；可见 token 使用 nearest-code 与 straight-through 梯度。
3. 目标损失为原任务损失 + `1.0 × VQ loss + 0.01 × MoE balance loss`；VQ loss 包含 codebook loss 和 `0.25 × commitment loss`。各系数可配置。
4. 码本和 memory bank 随模型 state_dict 保存，也能单独导出。bank 是最终权重对训练集无随机遮挡遍历后激活的码本项，保存向量、索引和计数；不使用验证/测试集或训练期间历史命中代替最终遍历。
5. 常用 pretraining、reconstruction、embedding、forecast 路径采用同一输入增强。支持关闭 MoE/VQ 的消融开关。首版增强结构支持 T5/Flan-T5 encoder-only，关闭 MoE 的梯度检查点以保证辅助损失梯度正确。
6. 提供单卡 CPU/CUDA 云训练入口、CSV/NPY 数据、离线小模型验证、检查点恢复和独立 bank 重建。先按时间切分数据，再用训练段拟合缩放并滑窗。
7. 修正原预训练循环停止条件、单 GPU 外部评估假设和下游 checkpoint 加载的 buffer 保留。原研究入口仍需其原有 PILE/W&B 依赖；轻量入口不依赖这些服务。

## 方法边界

这是 COMET 启发的输入 VQ 瓶颈。按用户选择，VQ 位于 T5 之前，与 COMET 对上下文编码结果量化的顺序不同；不宣称完整复现 COMET 的多尺度、异常评分、最优传输或在线适应。

云主配置使用离线随机初始化的较小 T5，便于上传即运行；不是已训练好的 MOMENT 基础模型。提供预训练 T5 和现有研究版 checkpoint 的迁移接口，新增路由与 VQ 仍需训练。

本次验证包括单元测试和数步训练/恢复；不在未指定服务器、数据集与预算的情况下启动正式云端训练，不承诺下游精度提升。实际验证结果见交付说明。

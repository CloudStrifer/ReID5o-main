# 实现对应关系与设计决定

当前架构版本：`mfcraf-v2`。原代码备份：`output/pre_alignment_backup/`。旧实验权重不兼容新模型，加载器明确拒绝静默迁移。

| 原核查项 | 现实现 |
|---|---|
| SPA-LoRA 不存在 | `model/spa_lora.py`：按图像高度划分 P 个 patch 区域，每区 E 个低秩专家；逐样本完整 v 经独立路由 softmax；插入视觉 Transformer 最后若干层 MLP 的两个线性映射。保留预训练注意力头数，不再重建错误的头数。 |
| 主检索路未接 CMC/RAF | `model/build.py::encode_multimodal` 为唯一编码入口；其输出用于 ID、SDM、一致性、测试和展示。即使两个补全损失权重均为零，检索损失仍训练生成器。 |
| 假特征充当补全目标 | 完整真实可用输入经无梯度 teacher 分支提取监督；仅对“确实存在且本次人工遮挡”的目标计算 reconstruction。天然不存在的 NIR/CP 不充当监督目标。 |
| 全 batch 共用掩码／三模态全空 | `ModalityDropout` 对 [B,5] 掩码逐样本采样，先限制真实可用模态，再修复保留数；最少保留数按各样本可用数截断。 |
| 缺失 token 未真正学习 | MRE 缺失序列均值、type 与 mask 嵌入作为目标模态固定语义查询的先验；与生成器 learnable query 相加。它们参与主检索梯度。被遮挡的文本内容及 EOT 位置不参与该先验。 |
| RAF 权重被覆盖 | 删除未加权 Transformer 覆盖分支，直接输出 Proj(sum(w*u))；评分网络恰为两层 Linear；置信度为一层 Linear+sigmoid；温度固定且可配置。 |
| uncertainty 公式不同 | 使用 sum_k w_k [g_k + relu(delta-s_k)]，生成标志逐样本传递，无额外 0.5 系数或对 s 的 sigmoid。 |
| consistency 解释错误 | 对完整融合表示与缺失融合表示计算 mean(1-cos)，teacher 停梯度。删除没有作用的温度参数。 |
| 测试和展示不同 | `test.py`、`infer.py`、特征导出共用同一 encoder，删除固定 0.7/0.3 混合与异常静默回退。 |
| 测试缺失率无实现 | `utils/protocol.py`：按版本、seed、任务名、样本索引生成可重放 mask，导出实际删除率与逐样本记录。 |
| 默认配置不对齐／消融不能关闭 | 默认启用 SPA/MA/CC/RF、mAP、30 epochs；所有开关支持 `--no-...`；配置文件及 CLI 覆盖均可用。 |
| 测试集挑 best | 只使用正式 val 选择 best；没有 val 的 ICFG/ORBench 固定最后一轮，best.pth 仅是 final 的别名。拒绝以 test 做 validation。 |
| CUHK 错误划分与素描配对 | 显式 split 字段或正式 manifest；按完整相对路径 stem 配对，兼容 BMP/JPG；缺失 SK 标记 unavailable 或报错。无 RGB 回退。 |
| 重写遗漏随机配对 | 恢复三模态数据的多 caption 候选和 ORBench 同身份辅助模态池，每轮训练重新采样。 |
| 重建与循环的平均分母不一致 | 两项均先在样本内平均，再平均完整 batch；循环集合按样本抽取；空集合贡献零属于明确的边界约定。 |
| 类别数写死 | checkpoint 保存真实训练类别数，测试/展示恢复该值并严格检查。 |
| 指标可能 OOM／mAP=0 误导 | 按 query 分块计算排序，内存受 chunk_size×gallery_size 限制；没开启 mAP 时返回 null；缺少相关 gallery 的 query 显式报错。 |
| 复现记录缺失 | 保存 configs、源代码 SHA256、环境、训练损失、checkpoint、评估 checkpoint SHA256、seed、缺失 mask、五模态平均权重。 |
| 分词器擅改 CLIP 词表 | 恢复原始 CLIP 词表，移除原训练路径未在论文交代的随机 MLM 遮词。 |
| 模块／损失／参数消融脚本缺失 | `tools/run_experiments.py` 支持 full、ablation、sensitivity、spa、rgb，默认只生成可审阅计划，显式 --execute 后顺序训练与评估。 |

补充约定：

- 论文 E/P 自相矛盾；当前默认 E=4、P=3，依据图 9 和敏感性分析。两个值均可设置，未把旧表当作新实现验证。
- MRE 使用 pooled missing sequence 组成模态专属语义查询，保证该先验实际参与补全。所有生成上下文都来自观测模态；不可用输入不能泄漏到表示。
- 三模态数据仅激活 RGB/SK/TEXT；模型仍统一用五列状态表示，未激活项权重严格为零。
- 每个样本默认随机采样最多 3 个有效有向模态对做 cycle，可通过 `cycle_pairs_per_step` 调整，RNG 状态随 checkpoint 保存。当前源和中间模态均要求真实可用；这项限制尚需与论文定义核实。
- ID 对查询和 RGB gallery 最终融合表示的 CE 取平均；SDM 在两个融合表示之间计算。取消旧版对 16 个含缺失占位特征的组别损失。
- MRE/CC/RF 的原 warmup 仍保留在训练器中。日志记录实际生效的损失权重结果。
- 视觉基础权重默认冻结；SPA 参数可训练；文本默认可训练，另有冻结开关。

本表说明现有实现，不构成“论文全部补齐”的验收结论。具体假设与未完成项见 [07 忠实度再核查](07_fidelity_reaudit.md)。


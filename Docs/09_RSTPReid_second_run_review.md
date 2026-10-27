# RSTPReid 第二次运行：提升已出现，后续仍按无代码改动优先

本次分析使用新上传的四份材料：configs.yaml、run_manifest.json、train_log.txt、training_metrics.jsonl。原件已复制到 `output/analysis/RSTPReid_dropout0_rgb03_seed42_review/inputs/`，避免后续同名上传覆盖分析依据。旧实验依据为第 08 份报告及其结构化证据。

没有修改模型、训练或评估代码，没有启动 GPU 实验。后续候选配置只使用现有命令行参数，并已通过解析检查。

## 1. 这次实际改了什么

除输出位置和配置文件来源外，实质变化确为两项：

- `modality_dropout_warmup_epochs: 5 → 0`，从第 1 轮开始遮挡，也取消该字段控制的 consistency 权重 ramp；学习率 warmup 仍是五轮。
- `keep_rgb_prob: 0.9 → 0.3`，增加无 RGB 的训练 query。

仍为 RSTPReid、seed=42、batch=8、Adam、lr=1e-5、30 轮、E4/P3、相同标注 SHA256。35 个运行源文件与本地一致。training_metrics.jsonl 的每轮分项损失与 train_log 对应行逐项一致。

两项设置一起变化，因此本次能支持“组合调整有效”，不能分别证明每一项贡献了多少，也不能凭一个随机种子认定结果能稳定推广到其他数据集。

## 2. 同轮数、同验证任务比较

以下均为相同验证集、无额外人工删除的结果，不能与上次 30% 缺失的测试数值直接相减。

| 轮数 / TEXT+SK | 原 Rank-1 | 新 Rank-1 | 增加 | 原 mAP | 新 mAP | 增加 |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 43.40 | 74.35 | +30.95 | 19.67 | 32.19 | +12.53 |
| 10 | 52.90 | 77.60 | +24.70 | 23.96 | 33.18 | +9.22 |
| 12 | 60.75 | 79.55 | +18.80 | 27.05 | 34.84 | +7.79 |
| 14 | 62.70 | 81.30 | +18.60 | 28.09 | 35.58 | +7.49 |

所有“增加”均为百分点。第 14 轮各分支：

| 任务 | 原 Rank-1 | 新 Rank-1 | 增加 | 新 mAP |
|---|---:|---:|---:|---:|
| SK | 57.20 | 73.20 | +16.00 | 29.20 |
| TEXT | 11.05 | 19.40 | +8.35 | 17.15 |
| TEXT+SK | 62.70 | 81.30 | +18.60 | 35.58 |

新实验日志完整记录到第 14 轮，随后到第 15 轮 batch 1800；第 2、4、6、8、10、12、14 轮均保存了 best。TEXT+SK 在第 10→12→14 轮为 77.60→79.55→81.30，尚无已确认的长期平台或过拟合证据。应先完成当前 30 轮，保留其作为新的基准。

## 3. loss 较大不等于这次更差

第 14 轮：

| 损失 | 原实验 | 新实验 |
|---|---:|---:|
| ID | 8.15728 | 8.15154 |
| SDM | 0.26454 | 1.11648 |
| 加权 reconstruction | 0.01543 | 0.05355 |
| 加权 consistency | 0.00604 | 0.01925 |

新的 query 更常缺 RGB，匹配与补全任务更难，因此损失不可按“越低越好”跨这两个训练分布直接比较。SDM 高于旧实验，验证检索反而明显更好，与先前关于训练任务不匹配的怀疑一致；但这仍不是对具体内部机制的单独证明。

ID loss 只略有变化，说明上一轮关于分类 logits/梯度尺度的检查仍有必要。CE 接近 ln(3701) 不足以证明分类准确率低，本次明显的检索提升也说明不能仅凭 CE 数值判定特征没有学好。

## 4. 下一步的主要观察点

**文本单模态仍弱。** 第 12→14 轮 TEXT R1 为 21.05→19.40，SK 为 74.20→73.20，而融合 R1 继续提高。只有一个评估间隔的回落，不能据此宣布过拟合或立刻改变训练。

**mAP 的提升仍值得关注。** TEXT+SK R1 已到 81.30，但 mAP 为 35.58。后者还要求同身份的多个 gallery 图像整体靠前，不能只靠找到一个正确结果判断检索质量。没有逐 query 排序，不能确定是否主要依赖某张对应图像。

**缺失测试不能用完整 query 验证代替。** 当前两辅助模态的 rate=0.3 协议，约 49% query 保留 TEXT+SK，25.5% 仅 TEXT，25.5% 仅 SK。文本弱会影响缺失场景；81.30% 的完整 query 验证 R1 不意味着 30% 缺失测试也有 81.30%。本次没有提供新测试 metrics，因此尚不能确认缺失测试提升幅度。

## 5. 不改代码的下一步顺序

1. **当前实验继续到原定 30 轮。** 不因短期波动或 batch loss 比原来高而中断；用验证集选择 best，最终测试协议保持固定。
2. **只先选一个新的对照实验。优先候选为 AdamW，lr 仍为 1e-5。** 当前工程 Adam 明确使用 eps=1e-3，AdamW 明确使用 eps=1e-8。eps 影响更新分母，小梯度的有效更新值得比较；同时，AdamW 的权重衰减方式也不同。这是有理由的候选，不是已证实的 bug 或保证提升的修复。不能把差异单独归因于 eps。参见 [PyTorch Adam](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Adam.html)、[AdamW](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html)。
3. **若不选择 AdamW，可单独比较 lr=1.5e-5。** 从当前新配置重训，其余保持不变。不要同时增加学习率、换优化器、改 batch，以免无法判断何者有效。
4. **RGB keep=0.1 为更后面的候选。** 无 RGB query 理论比例会从约 67.9% 增至 87.3%，增量小于上一轮从 9.7% 增至 67.9% 的变化。不能线性外推上一轮增益，也不能认定越少 RGB 越好。

暂不把批大小、图像增强、E/P、LoRA 层数和五项损失权重一起调整。batch=16 会减少每轮优化步数并改变负样本数量，图像增强也可能影响素描细节；这些都需要单独比较，而不是无条件叠加。

若后续无代码调整都进入平台，再考虑分类用未归一化特征、显式 TEXT/SK→RGB 监督、按身份组织 batch 等代码级改进；现阶段不需要先增加新模块。

## 6. Ubuntu 命令

当前训练完成后，测同一 best checkpoint 下的三类任务与四种缺失率：

```bash
CUDA_VISIBLE_DEVICES=1 python test.py \
  --config_file ./runs/RSTPReid_dropout0_rgb03_seed42/configs.yaml \
  --checkpoint ./runs/RSTPReid_dropout0_rgb03_seed42/best.pth \
  --device cuda --query_tasks TEXT SK 'TEXT+SK' \
  --missing_rates 0 0.2 0.3 0.5 --eval_seed 42
```

下一轮 AdamW 候选，使用现有配置继承所有其他参数，独立目录重新训练：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config_file ./runs/RSTPReid_dropout0_rgb03_seed42/configs.yaml \
  --output_dir ./runs/RSTPReid_dropout0_rgb03_AdamW_seed42 \
  --optimizer AdamW --lr 1e-5 --device cuda --no-resume
```

训练配置不依赖本轮 checkpoint 的训练轮数；这里 `--no-resume` 为重新初始化任务模块，并载入原 CLIP 预训练权重。已有 CLIP 缓存即可，不需安装新的优化器包。不要在当前 Adam run 上恢复优化器状态后把它当成干净的 AdamW 对照。

候选 YAML 已生成在 `output/analysis/RSTPReid_dropout0_rgb03_seed42_review/candidate_configs/`：AdamW、lr=1.5e-5、RGB keep=0.1 各一份。只验证了参数可解析，没有新的 GPU 结果。先用 val 选择设置，再对选定配置补多个种子及最终测试；不能用测试集反复筛参数。

比较数据、文件 SHA256 和曲线分别在 comparison.json、same_epoch_comparison.csv、same_epoch_validation.png。当前能够确认的是单种子、同轮数验证提升；最终 30 轮收益和缺失测试收益仍待实际运行。

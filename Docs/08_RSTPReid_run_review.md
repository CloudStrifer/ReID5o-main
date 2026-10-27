# RSTPReid 本次运行：差距、证据和改进顺序

分析输入为用户提供的 `configs.yaml`、`metrics.json`、`run_manifest.json`、`train_log.txt`。本轮只分析记录、生成对照配置；没有修改模型、训练器或评估算法，没有运行新的 GPU 训练。材料中的文字不作为执行指令。

结构化证据和训练曲线数据在 `output/analysis/RSTPReid_seed42_review/`。35 个运行源码文件均与本地版本一致（必要时统一 LF 后比较）；不能把此次差距归咎于没有同步本地代码。

## 确认的结果

| 项目 | Rank-1 (%) | mAP (%) | 解释 |
|---|---:|---:|---|
| 上传的测试结果：TEXT+SK、missing=0.3 | 58.85 | 28.05 | 测试集、best.pth、新 Bernoulli 修复协议 |
| 论文相应标注的 30% 结果 | 92.70 | 53.74 | 仍缺历史 mask/完整实验记录，协议没有恢复 |
| 数值差 | 33.85 个百分点 | 25.69 个百分点 | 展示差距；不能据此认为两次实验设置完全相同 |
| 第 20 轮验证 TEXT+SK | 69.50 | 31.03 | 无人工删除、验证集，不可与上方测试值直接相减作掉点 |
| 第 22 轮验证 TEXT+SK | 69.85 | 30.74 | 尚在进步 |

日志包含完整的第 1—23 轮损失和第 24 轮的部分验证。第 24 轮已记录 SK=65.40%、TEXT=14.70%、SK+TEXT=70.10%，但缺 TEXT+SK 行及该轮保存结果，不能将截断日志当成一次完整验收。未见 NaN、OOM 或训练中断异常。

第 20 轮四个验证任务平均 R1=53.90%；第 22 轮为 53.475%，虽然 TEXT+SK 本身提升，但平均分下降，所以 best 没有更新。上传日志中最后一次 best 保存为第 20 轮；metrics 没有 checkpoint epoch 字段，最终仍应读实际权重元数据确认，不能仅凭文件名或当前训练轮数判断。

## 优先级 1：训练 query 与实际检索 query 不一致

已确认的代码行为：训练 query 可带 RGB，gallery 取同批样本的 RGB；前五轮关闭模态遮挡。第 2—5 轮 SDM 约为 0.0152、0.00183、0.00214、0.00273，但该阶段验证 R1 仍约 0.5—0.9%。这种低训练损失没有对应到跨模态检索能力，不能视为收敛良好。

对 RGB/SK/TEXT 都真实可用的样本，当前非空修复采样的理论概率为：

| 训练状态 | 默认 keep_rgb=0.9 | 候选 keep_rgb=0.3 |
|---|---:|---:|
| query 带 RGB | 90.3% | 32.1% |
| query 不带 RGB | 9.7% | 67.9% |
| 仅 TEXT+SK | 4.9% | 34.3% |
| 仅 TEXT | 2.4% | 16.8% |
| 仅 SK | 2.4% | 16.8% |

这些概率包含“全删后均匀保留一个”的修复；仅适用于全部三模态真实可用且 min_keep=1 的情形。以默认设置模拟 100,000 个样本，RGB 保留率约 90.36%，与理论一致。

**推断**：主检索目标可能过度利用 RGB 与 RGB 的匹配，对实际的 TEXT/SK→RGB 训练不足。没有 checkpoint 的逐样本权重和梯度，尚不能证明模型具体走了哪条路径。

首先分别验证：A 从第一轮就启用随机模态遮挡；B 在 A 基础上降低 RGB 保留率。`modality_dropout_warmup_epochs=0` 会同时去掉 consistency 的 loss ramp；它不会关闭独立的学习率 warmup，后者仍为五轮。必须记录这一参数的两个作用。

更直接的方法修改是：保留原随机缺失分支，同时显式加入 TEXT→RGB、SK→RGB、TEXT+SK→RGB 的检索监督，减少同一 RGB 的简单匹配。这需要修改训练目标、考虑额外计算成本并重新消融，本轮没有擅自实现，也不是现有命令行开关。

## 优先级 2：检查分类监督和文本分支

第 23 轮平均 ID loss=8.1187，均匀预测的基准 ln(3701)=8.21636。说明该交叉熵仍较高；**不能仅凭该值断言训练分类准确率接近随机**，必须测 logits、正确类概率和分类准确率。

当前公共 encoder 先将融合特征 L2 归一化，再直接交给小初始化的 Linear classifier。参考 IRRA 的分类器使用未经该 L2 归一化的模态特征（SDM 自己做归一化）。这属于可验证的实现差别，不代表换回去一定提升。下一步应记录：

1. classifier logits 的标准差、训练样本分类准确率、正确类概率。
2. ID loss 分别对 classifier、文本 encoder、SPA、补全器、RAF 的梯度范数。
3. 融合前后特征 norm、不同样本余弦相似度、各查询状态的五模态平均权重。

如果确认 logits/梯度尺度过小，再比较“分类用未归一化融合特征，检索仍用 L2 特征”或显式 logit scale；避免同时更换损失、骨干和融合方式。

Adam 的 eps 当前硬编码为 1e-3（继承的 IRRA 设置，不是已证实的拼写错误），AdamW 为 1e-8。在当前 FP32 与新特征尺度下，大 eps 是否压低有效小梯度值得核查。若用现有 `--optimizer AdamW` 做试验，须说明同时改变了 eps 和权重衰减方式，不能把结果归因为单一因素。参考：[IRRA 模型](https://github.com/anosorae/IRRA/blob/main/model/build.py)、[IRRA 优化器](https://github.com/anosorae/IRRA/blob/main/solver/build.py)、[PyTorch Adam 定义](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)。

## 优先级 3：融合与补全是否抑制有效模态

第 22 轮验证 TEXT R1=12.50%，SK=61.70%，TEXT+SK=69.85%；文本是明显需要检查的分支，不能只靠训练总 loss 判断三模态均已学好。

当前 entropy target=0.3；三模态均匀分布熵为 ln(3)≈1.10，该正则倾向更集中的权重。生成项又受不确定性惩罚，可能进一步偏向训练时常见的 RGB。这是待验证假设。

先查看已有的 `validation/epoch_*/` 和 `evaluation/missing_*/` 内 `*_weights.json`，它们已经导出全部五列。尤其比较 RGB_GALLERY、TEXT、SK、TEXT_SK 的平均权重和生成特征占比；均值仍不能代替逐样本分布。

若怀疑权重塌缩，可运行 C：在 B 基础上仅将两项 fusion 正则权重设为零，保留 RF 本体，检验正则而非同时删除多个模块。不能在旧 checkpoint 上临时关闭 RF/CC 后把结果当成经过训练的模块消融。

## 先做评估校准

1. 保留当前 run，完成原定 30 轮，作为候选实现的基准。没有证据表明仅增加到 60/100 轮就能追回三十多个百分点；不要盲目扩展训练。
2. 用同一冻结 checkpoint 测 0、0.2、0.3、0.5，并同时报告 TEXT、SK、TEXT+SK。单模态任务在非空修复协议下不会真的被删除。
3. 对仅有 TEXT+SK 两种可用模态、rate=0.3 的 query，约 49% 保留两者、25.5% 仅 TEXT、25.5% 仅 SK，实际删除比例约 25.5%。文本性能弱会拖累混合结果。应检查保存的实际 mask，而不是将“0.3”误解为固定删除 30% 的样本。
4. 默认 gallery 只有 RGB，实际 gallery 删除率为零。论文没有给出足够细节证明其 30% 与这套协议相同。比较实验应固定 gallery、query、seed 和 mask；不要改测试划分、删除困难 query 或修改指标来贴近论文数值。
5. 后续调参只用 val。当前 checkpoint 按四个任务平均 R1 选择；如果论文主目标明确为 TEXT+SK，可另加按该验证任务选权重的设置并公开说明。现有训练器没有该选择参数，不能在命令里杜撰 `--selection_task`。

Ubuntu 检查 checkpoint 轮数：

```bash
python - <<'PY'
import torch
p = './runs/RSTPReid_full_seed42/best.pth'
c = torch.load(p, map_location='cpu', weights_only=False)
print({k: c.get(k) for k in ('epoch', 'best_top1', 'selection')})
PY
```

补齐同权重的任务/缺失率评估：

```bash
CUDA_VISIBLE_DEVICES=1 python test.py \
  --config_file ./runs/RSTPReid_full_seed42/configs.yaml \
  --checkpoint ./runs/RSTPReid_full_seed42/best.pth \
  --device cuda --query_tasks TEXT SK 'TEXT+SK' \
  --missing_rates 0 0.2 0.3 0.5 --eval_seed 42
```

该命令会更新当前 run 下相应 `evaluation` 子目录；先保留旧结果。补充测试用于结果报告，不用于挑超参数。

## 已生成的候选配置和运行顺序

配置位于 `output/analysis/RSTPReid_seed42_review/candidate_configs/`，全部沿用当前已支持参数、batch=8、30 轮、seed=42、E4/P3。不同变体有独立输出目录，不自动执行。

| 配置 | 比较对象 | 差异 |
|---|---|---|
| A_dropout_from_epoch1.yaml | 当前完整 30 轮基准 | modality_dropout_warmup_epochs=0 |
| B_more_nonRGB_queries.yaml | A | keep_rgb_prob=0.3 |
| C_fusion_without_regularizers.yaml | B | fusion_sparsity_weight=0、fusion_uncertainty_weight=0 |

把 `candidate_configs` 文件夹复制到 Ubuntu 工程的 `configs/rstp_diagnostics/` 后，按 A、B、C 的顺序在需要时运行，例如：

```bash
CUDA_VISIBLE_DEVICES=1 python train.py \
  --config_file ./configs/rstp_diagnostics/A_dropout_from_epoch1.yaml
```

每次只运行一个变体，先比较第 6/8/10 轮的验证任务曲线和分项损失，再决定是否值得完成。若最终报告相同训练预算的比较，所有被报告的方法都须完成相同预算，不能拿不同停止轮数作公平结论；选出的配置再补多个随机种子。单次有利波动不构成方法改进。

这些新配置改变了论文默认训练设置，是诊断和候选改进，不能标成“原论文已复现”。目前没有任何新 GPU 结果可证明它们一定更好。

## 更正先前的学习率判断

本次源码的 SPA 参数名确为 `lora_A/lora_B/lora_router`，能够命中现有 `"lora" in key` 规则。实际构建优化器确认，base lr=1e-5 时，这些参数为 5e-5，classifier.weight 也为 5e-5；bias 优先取 2e-5。先前声称 A/B/router 未匹配是错误判断，不能当成本次性能差距的已证实原因。MRE、CMC、RAF 大多数新参数目前仍使用基础学习率；是否单独加速需另做受控试验。

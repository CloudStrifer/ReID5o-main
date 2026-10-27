# 验证结果与剩余实验

功能验证在本机 CPU 环境执行。本轮结果：**32 项测试通过，36.15 秒。** 上轮 28 项通过只代表当时的检查范围，未能发现的遗漏现已补充检查。静态未定义符号检查通过；pip check 无依赖冲突。当前自动测试覆盖：

- 真实可用性约束、逐样本 dropout、至少一个观测、三模态 NIR/CP 不复活。
- SPA 空间分块、状态路由、冻结 W0、路由梯度、无 SPA 分支。
- 无 reconstruction/cycle 辅助损失时，补全器、可靠性评分器和投影仍从检索目标获得非零梯度。
- MRE 缺失序列和 mask=0 嵌入确实收到检索梯度。
- 修改已遮挡图像或文本内容/EOT 位置不改变表示。
- 三模态的无效模态权重为零，不产生对应补全监督或生成器梯度。
- RAF 权重影响输出；uncertainty 数值与公式一致。
- 独立手算重建与循环损失：不同样本有不同目标数/循环对数，空集合贡献零，按完整 batch 平均。
- 每轮随机 caption 与同身份模态采样可重放、不修改原始记录，训练循环确实调用重采样。
- 确定性 mask 与 batch 顺序无关；rate=1 修复非空；单模态保持可用。
- 小 gallery 的 Rank 指标、缺少匹配的处理、mAP 未计算时为 null。
- 分块排序与完整排序数值一致。
- CUHK 正式 split 要求、BMP/JPG 素描配对、缺失 SK 掩码/报错。
- 从现有 ZIP 只临时读取三个真实 RGB/素描配对做数据烟雾验证；测试夹具的划分不是正式 benchmark 划分。
- 原生 CLIP 类上的前向/反向（缩小网络），以及实际优化器训练步。
- checkpoint 类别数恢复、严格架构版本、旧权重拒绝。
- 配置文件与 CLI 覆盖、消融开关。
- Evaluator 与公共 encoder 的数值一致性。
- 子进程完整闭环：小型随机 CLIP 权重 → train.py → 保存 → test.py 两个缺失率 → infer.py 检索图 → export_features.py 成对特征。输入仅为测试夹具，不是论文实验。

运行命令：

```powershell
.\.venv\Scripts\python.exe -m pytest -q --junitxml=output/verification/junit.xml
```

验证环境与测试结果以最终 JUnit XML 和 validated_environment.txt 为准。

以下尚不属于“已完成实验”：

1. 原始预训练 ViT-B/16 在全部真实数据上的训练、GPU 显存/耗时。
2. 四数据集正式 Rank/mAP、消融、敏感性、E/P 与 RGB keep 实验。
3. 论文八种外部基线的复现及公平对比。
4. 实际 checkpoint 上的可靠性表、t-SNE 和检索示例更新。
5. CUHK 正式 split 文件仍未随当前数据包提供，需补齐后才可跑正式 CUHK 训练。

被覆盖的小规模行为和命令行闭环已用测试验证；不能以此声称达到论文中的已有数值，也不能证明所有实现选择忠实于论文。相关假设与遗漏见 [07 忠实度再核查](07_fidelity_reaudit.md)。旧实现保留在备份中，历史核查报告的代码链接已指向备份，避免旧行号误指新实现。


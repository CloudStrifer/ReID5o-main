# 运行与复现

## 当前工程的直接运行顺序（PowerShell）

当前路径为 `E:\Code\ReID5o-main`，数据根目录为其 `data` 子目录。RSTPReid 的读取检查得到 18,505 条训练记录、3,701 个训练身份、验证/测试各 1,000 张 gallery，缺失素描为零；首条 RGB/SK 路径存在。这不是全量图像解码验证。建议先跑该数据集。

本机 `.venv` 已核实为 CPU 版 PyTorch，`cuda_available=False`，可立即运行功能测试：

```powershell
Set-Location 'E:\Code\ReID5o-main'
.\.venv\Scripts\python.exe -m pytest -q
```

以下正式训练/测试命令中的 `python` 指向已安装 CUDA 版 PyTorch 的环境；先激活该环境，并确认下面的检查输出为 `True`。当前 CPU `.venv` 无法执行 `--device cuda`。本轮未安装或替换 GPU 环境。

```powershell
Set-Location 'E:\Code\ReID5o-main'
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

单次训练（当前候选实现；不等于已确认历史论文配置）：

```powershell
python train.py --dataset_name RSTPReid --root_dir 'E:/Code/ReID5o-main/data' --output_dir 'E:/Code/ReID5o-main/runs/RSTPReid_full_seed42' --device cuda --num_epoch 30 --batch_size 8 --test_batch_size 16 --num_workers 0 --seed 42 --spa_experts 4 --spa_parts 3 --log_period 20
```

默认开启 SPA、MRE、CMC、RAF 和 mAP；默认使用 ViT-B/16，首次会下载 CLIP 预训练权重。本条只训练一个随机种子；完整 ViT GPU 训练尚未由本轮实际验证，显存不足时应记录并调整运行设置。训练完成后，以正式 val 选出的 best.pth 测试四种 query 缺失率：

```powershell
python test.py --config_file 'E:/Code/ReID5o-main/runs/RSTPReid_full_seed42/configs.yaml' --checkpoint 'E:/Code/ReID5o-main/runs/RSTPReid_full_seed42/best.pth' --device cuda --query_tasks 'TEXT+SK' --missing_rates 0 0.2 0.3 0.5 --eval_seed 42
```

结果写入该 run 下的 `evaluation/missing_<rate>_seed_42/metrics.json`；同时保存缺失 mask 和可靠性权重。这里的缺失率按当前新协议执行，不代表已恢复原论文历史协议。

需要连续完成三个种子的完整方法训练和测试时，另用独立实验目录（默认 `experiments`）：

```powershell
python tools/run_experiments.py --dataset RSTPReid --root_dir 'E:/Code/ReID5o-main/data' --suite full --seeds 42 43 44 --execute
python tools/summarize_experiments.py --plan 'E:/Code/ReID5o-main/experiments/RSTPReid/full/plan.json' --output 'E:/Code/ReID5o-main/experiments/RSTPReid/full/summary.csv'
```

此批处理会重新训练 seed 42，不自动复用上面的手动运行。去掉 `--execute` 只生成计划；把 `--suite full` 改为 `--suite ablation` 则是 9 个变体 × 3 个种子，即 27 次训练。

其他数据集的前提：

- ICFG-PEDES：已解压，使用 `--dataset_name ICFG-PEDES`；无正式 val，按固定最后一轮选择权重。
- ORBench：当前只有 ZIP。先运行下面的解压命令，确认得到 `data/ORBench/train_annos.json`、`test_gallery_and_queries.json` 和 vis/nir/cp/sk，再将训练数据集名换为 ORBench，测试任务换为 `TEXT+NIR+CP+SK`。无正式 val，按固定最后一轮选择权重。
- CUHK-PEDES：还缺正式 split；补齐后加 `--split_file`，不能用现有目录名猜划分。

```powershell
Expand-Archive -LiteralPath 'E:\Code\ReID5o-main\data\ORBench.zip' -DestinationPath 'E:\Code\ReID5o-main\data'
```

## 环境

Python 3.10+。先在研究用 GPU 环境安装匹配的 PyTorch/torchvision，再安装依赖：

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

本机已建立 `.venv`，验证环境的完整包版本见 `validated_environment.txt`。该环境为 CPU 版 PyTorch，不代表论文 RTX-3090 训练环境。此虚拟机 oneDNN 内核不能运行，测试中关闭该后端；命令行 CPU 调试可加 `--disable_mkldnn --cpu_threads 1`。

首次实际使用默认 ViT-B/16 会按已有 CLIP 下载器获取预训练权重，也可将 `--pretrain_choice` 设为本地 CLIP 权重文件。此次命令行测试使用小型随机 CLIP 夹具，不冒称预训练 ViT-B/16 实验。

## 单次训练

```powershell
python train.py --dataset_name ORBench --root_dir E:/Datasets --output_dir runs/ORBench_full --seed 42
```

CUHK 增加 `--split_file 正式划分文件.json`。参数既能从 `--config_file 配置.yaml` 读取，也能由 CLI 覆盖。训练保存真实类别数、架构版本、源代码 hash、环境、损失日志与 checkpoint。

消融示例：

```powershell
python train.py --no-use_cross_modal_completion --no-use_reliability_fusion --output_dir runs/MA_only
python train.py --completion_recon_loss_weight 0 --completion_cycle_loss_weight 0 --output_dir runs/no_completion_losses
python train.py --spa_experts 4 --spa_parts 3 --output_dir runs/SPA_E4_P3
```

第一条移除模块；第二条只去掉补全辅助监督，生成器仍由检索目标训练。这两个实验含义不同。

## 冻结权重后的测试

```powershell
python test.py --config_file runs/ORBench_full/configs.yaml --missing_rates 0 0.2 0.3 0.5 --query_tasks TEXT+NIR+CP+SK
```

每个缺失率的结果、mask、平均权重写入 run 目录的 evaluation 子目录。测试始终加载同一个 checkpoint，不随缺失率重训。多模态 gallery 可加 `--gallery_file 文件.json --gallery_missing_rate 0.3`。数据原始目录更换时可修改副本配置的 root_dir，保留原始配置和结果。

## 顺序实验计划

```powershell
python tools/run_experiments.py --dataset ORBench --root_dir E:/Datasets --suite ablation --seeds 42 43 44
```

默认只生成 YAML 和 plan.json，不启动昂贵训练。检查计划及数据后，加入 `--execute` 才按顺序逐个训练和评估。suite 还支持 full、sensitivity、spa、rgb。CUHK 需指定正式 split_file。

已生成一个可审阅的 9 项 ORBench 消融示例计划：`output/experiment_plans/ORBench/ablation/plan.json`。此计划没有运行正式训练，summary 中缺失项记录为 pending，不会填入论文旧数值。

```powershell
python tools/summarize_experiments.py --plan experiments/ORBench/ablation/plan.json --output experiments/ORBench/ablation/summary.csv
```

汇总只读取真实生成的 metrics，输出均值、样本标准差及种子数。缺失实验单列 pending；一次运行不输出虚假的标准差。

## 特征与检索图

```powershell
python tools/export_features.py --config_file runs/ORBench_full/configs.yaml --task TEXT+SK --mask_modalities SK --missing_rate 0 --output runs/ORBench_full/completion_features.npz
python infer.py --config_file runs/ORBench_full/configs.yaml --sk_path E:/Datasets/query.jpg --text "A person wearing a blue coat." --output retrieval.png
```

导出包含最终表示、pid、观测 mask、权重；仅对确实存在而被人工遮挡的模态导出 real/completed 成对特征。没有真实目标的数据不伪装成可做重建对比的样本。可供后续 t-SNE 使用，图中应报告采样身份、seed、超参数和 checkpoint。

## 权重兼容

新方法应重新训练。旧 ReID5o 权重不是新架构的可复现结果；加载器不会通过忽略键名或维度来悄悄使用它。若需要预训练迁移，应单独实现并记录迁移清单，不可当成完整 MFCRAF checkpoint。


# 数据与评估协议

## 数据准备

数据可在任意磁盘，通过 `--root_dir` 指定。无需纳入版本控制。数据根目录通常为：

```text
data/
  ORBench/{vis,nir,cp,sk,train_annos.json,test_gallery_and_queries.json}
  CUHK-PEDES/{imgs,caption_all.json,reid_raw.json}
  ICFG-PEDES/{imgs,ICFG-PEDES.json}
  RSTPReid/{imgs,data_captions.json}
  sketch/aliyun/{CUHK,ICFG,RSTPReid}/imgs/
```

读取器也识别解压工具多加的一层目录、ICFG 压缩包外层名称以及素描包外层目录。根目录中同时存在多个标注候选会报错，避免选错版本。

CUHK 当前 caption_all.json 没有正式 split。必须补正式清单或带 split 的 reid_raw.json；不会再按 train_query/test_query 这些图像来源名猜测。manifest 支持：

```json
{"cam_a/000_45.bmp": "train", "cam_a/001_45.bmp": "test"}
```

上面只展示格式，不是真实划分。正式清单须覆盖全部标注。也可提供包含 file_path、id、split 的列表，或让 caption_all.json 本身包含 split。训练/验证/测试身份必须互斥。

素描按完整相对路径去扩展名后匹配；相同 stem 多个候选报错。缺少一张素描默认标记不可用，`--missing_sketch_policy error` 可改为报错；整个素描目录缺失直接报错。不会让 RGB 冒充 SK。没有真实输入的张量槽位为零，且不会进入有效观测或监督目标。

## 固定缺失协议 bernoulli-repair-v1

1. 区分数据集结构模态、样本真实可用性、人工删除后的观测状态。删除只发生在样本实际提供的模态上。
2. 使用 seed、任务名、样本索引的 SHA256 派生独立 RNG。mask 与 batch 大小、分批方式无关；不同方法使用相同样本顺序、任务名、seed 即共用协议。
3. 每个可用模态独立以 missing_rate 删除；如果全被删除，从原来可用的模态中均匀保留一个。
4. 导出 requested_rate、actual_deleted_fraction、available、observed、pid、index。requested_rate 是删除前 Bernoulli 概率；非空修复会降低实际删除比例，必须一起报告。
5. 单模态 query/gallery 实际删除率必为 0，因此单模态结果在相同 checkpoint 下不会因为请求删除率变化而变化。不能将它描述为单模态输入被删除后的鲁棒性。
6. 默认 gallery 只有 RGB。要研究 gallery 缺失，须提供多个真实 gallery 模态；仅 RGB 时不能同时删除 RGB 并要求至少一个观测。
7. 新协议为明确定义的实现选择，历史论文表格用的协议未知，不能宣称两者相同。若历史实验要求“精确比例删除”，须另做有明确分母与非空约束的协议，不能混用这套 Bernoulli 结果。

多模态 gallery 用 `test.py --gallery_file 路径.json`，格式：

```json
{
  "modalities": ["RGB", "SK", "TEXT"],
  "samples": [
    [123, "rgb/123.jpg", "sketch/123.jpg", "A person wearing a blue coat."],
    [124, "rgb/124.jpg", null, "A person carrying a bag."]
  ]
}
```

这是格式示例；pid 必须属于实际测试集，图像路径相对 manifest 目录或用绝对路径。值为 null 表示天然缺失；每条记录至少一个有效模态。对应真实数据才能运行。

## 报告与模型选择

- 验证只取正式 val；没有 val 就固定 final epoch，禁止用 test 指标挑模型。
- 主表任务在实验脚本中明确为三模态 TEXT+SK、ORBench TEXT+NIR+CP+SK。各查询方向可能有不同样本集合，不能把 CP+TEXT 与 TEXT+CP 默认视为同一评估集。
- 可靠性导出五列 RGB/NIR/CP/SK/TEXT，总和为 1；三模态未激活项为 0。不能无说明地隐藏 RGB 后仍声称四列是原始权重。
- 未计算 mAP 时 JSON 为 null。分块排序计算所有 gallery 的精确 mAP，不用 top-k 截断替代 mAP。


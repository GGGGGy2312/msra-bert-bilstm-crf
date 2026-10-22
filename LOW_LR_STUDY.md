# 小学习率继续训练：Mini 命名实体识别

## 训练设计

以此前选出的 UER Mini 权重插值检查点为起点，从其原有 40000 句训练行中使用固定种子 2043 抽取 10000 句，继续训练 **1 轮**。BERT 学习率为 `3e-6`，BiLSTM、分类层与 CRF 学习率为 `1e-4`；batch size 为 16，CPU 运行。开发集保持原来的 1024 句，训练阶段设置 `--test-size 0`，不读取测试成绩做模型选择。此前模型的权重和元数据备份在本机 `.cache/mini_before_low_lr/`，候选保存在 `.cache/mini_low_lr_10k/`。

| 模型 | 同一开发集实体级 F1 |
| --- | ---: |
| 上一版权重插值模型 | 93.955% |
| **小学习率继续训练后** | **94.168%** |

开发集确认提升后，才进行以下两项同集对照。实体类别、开始位置和结束位置全部一致才记为正确。

## 留出样本：主要新增证据

从 MSRA `train.tsv` 中筛出未用于上一版模型训练和开发的 1529 句。先排除此前词表方案审计抽过的 500 句，再用种子 2050 抽取 **500 句**；这些句子也不在本轮微调的 10000 句内。

| 模型 | 正确 | 误报 | 漏报 | Precision | Recall | 实体级 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 上一版 | 739 | 50 | 46 | 93.66% | 94.14% | 93.90% |
| **本轮候选** | **740** | **42** | **45** | **94.63%** | **94.27%** | **94.45%** |

提升约 **0.55 个百分点**。逐类 F1：人名 96.85% → 96.55%（略降）；机构名 89.52% → 90.75%；地名 95.18% → 95.69%。这是从训练文件剩余行抽取的留出集，**不是** MSRA 官方测试集。按句配对、重复 5000 次的 bootstrap 百分位 95% 区间约为 **−0.08 至 +1.27 个百分点**，包含 0；当前样本上的提升已观察到，但尚不足以断言对更大数据总体也有提升。

## 复用测试抽样：补充核验

在上一轮已公开使用的 500 句 MSRA `test.tsv` 抽样上，上一版和候选再次做相同行号对照：

| 模型 | 正确 | 误报 | 漏报 | 实体级 F1 |
| --- | ---: | ---: | ---: | ---: |
| 上一版 | 559 | 59 | 61 | 90.31% |
| **本轮候选** | **560** | **52** | **60** | **90.91%** |

提升约 **0.60 个百分点**。这批测试句之前已用于项目评估，属于**复用测试抽样**，不能声称是新的盲测结果，也不能据此推断完整 MSRA 测试集的分数。网页将两种数据来源分开标示。

## 复现

先按 [权重插值实验](BLEND_STUDY.md) 重建上一版权重，再备份到 `.cache/mini_before_low_lr/`。以下命令从上一版权重开始训练，并保留原开发行号：

```powershell
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --model mini --train-size 40000 --dev-size 1024 --test-size 0 --epochs 1 --batch-size 16 --threads 8 --seed 2029 --bert-lr 0.000003 --head-lr 0.0001 --init-checkpoint .cache\mini_before_low_lr\best.pt --init-run .cache\mini_before_low_lr\run.json --resume-subset-size 10000 --subset-seed 2043 --output-dir .cache\mini_low_lr_10k --report .cache\mini_low_lr_10k_report.md
.\.venv\Scripts\python.exe evaluate_low_lr_candidate.py holdout
.\.venv\Scripts\python.exe evaluate_low_lr_candidate.py reused
.\.venv\Scripts\python.exe bootstrap_low_lr.py
```

评估输出、固定抽样行号和权重哈希分别在 `experiments/low_lr_holdout.json`、`experiments/low_lr_reused.json`；重抽样区间在 `experiments/low_lr_bootstrap.json`，当前 Demo 的配置在 `experiments/low_lr_run.json`。权重文件仍只保存在本机；重建后可运行 `activate_low_lr_candidate.py`，脚本会检查开发集和两项同集对照均有提升，再激活候选。

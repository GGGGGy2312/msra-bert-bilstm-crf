# 按步评估与继续训练：第 125 步候选审计

## 结论

**当前 Demo 权重不变。** 在固定 1024 句开发集上，第 125 步候选 F1 从 94.168% 升至 94.327%；在此前未参与训练、开发或前两轮留出评估的 500 句 `train.tsv` 上，F1 从 93.782% 升至 93.909%。后一个差异只相当于多识别正确 **1 个实体**。在已经用过的 500 句 `test.tsv` 子集上，F1 从 90.909% 降到 90.511%。证据不足以说明候选稳定优于当前模型，因此没有替换线上 Demo。

## 改动与实验设计

- `bert_bilstm_crf.py` 新增 `--eval-steps N`。每 N 个训练步在**同一开发集**上评估，保存 F1 最高的检查点；`0` 保持原来的每轮末评估。每次评估后恢复训练模式，避免后续步骤错误地关闭 dropout。
- 从当前 Demo 检查点继续训练，训练行来自原 40000 句训练集的固定 10000 句子集，随机种子 2045；BERT 学习率 `3e-6`，LSTM/CRF 学习率 `1e-4`，batch 16，评估间隔 125 步。整轮结束为 625 步。
- 不使用预留样本选择步数。最佳步数 125 只由开发集决定；随后才读取预留 500 句作同集对照。

| 数据 / 用途 | 当前模型 F1 | 第 125 步候选 F1 | 差值 |
| --- | ---: | ---: | ---: |
| 开发集 1024 句，用于选择步数 | 94.168% | 94.327% | +0.159 个百分点 |
| 新预留的 `train.tsv` 500 句 | 93.782% | 93.909% | +0.127 个百分点 |
| 已使用过的 `test.tsv` 500 句，补充稳定性检查 | 90.909% | 90.511% | −0.398 个百分点 |

不按这三组不同样本的绝对 F1 排名。预留组的正确实体是 739→740，误报 42→41，漏报 56→55；测试组是 560→558，误报 52→55，漏报 60→62。测试组的复用历史意味着它不构成新盲测，但回退仍是拒绝替换权重的实际依据。

同学习率、同子集、仅在轮末评估时，F1 为 94.052%；另一组 10000 句、较低学习率的整轮结果为 94.149%。BIO 合法路径约束在开发集上未改变 F1。它们均未提高现有 Demo 的可靠性。

## 复现

已有本地权重和数据时，在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --model mini --train-size 40000 --dev-size 1024 --test-size 0 --epochs 1 --eval-steps 125 --batch-size 16 --threads 8 --seed 2029 --bert-lr 0.000003 --head-lr 0.0001 --init-checkpoint .cache\bert_bilstm_crf\best.pt --init-run .cache\mini_before_low_lr\run.json --resume-subset-size 10000 --subset-seed 2045 --output-dir .cache\mini_low_lr_round3_midpoint --report .cache\mini_low_lr_round3_midpoint_report.md
.\.venv\Scripts\python.exe evaluate_midpoint_candidate.py holdout
.\.venv\Scripts\python.exe evaluate_midpoint_candidate.py reused
.\.venv\Scripts\python.exe evaluate_bio_decode.py
```

抽样行号、指标和检查点 SHA-256 保存在 `experiments/midpoint_holdout.json`、`experiments/midpoint_reused.json`、`experiments/bio_decode_dev.json`。本机权重在 `.cache/`，不包含在 GitHub 仓库中；复现评估需要先生成对应检查点。由于可用的短句留出池已被多轮实验消耗，下次比较建议先明确新的数据划分或引入外部标注集，避免继续用同一批样本选择模型。

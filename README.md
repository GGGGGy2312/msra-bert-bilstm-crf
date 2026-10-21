# MSRA 中文 NER：BERT-BiLSTM-CRF

一个**独立**的中文命名实体识别学习仓库。用公开 MSRA NER 数据微调 UER 中文 BERT 结构模型、双向 LSTM 与线性链 CRF，识别人名、机构名、地名。仓库不依赖 LTP，也不包含数据集或预训练权重。

## 模型与数据

```text
输入字符 → UER Chinese RoBERTa-Mini（BERT 结构）→ BiLSTM → 线性层 → CRF → BIO 标签 → 实体跨度
```

- 当前 demo 的预训练模型：[uer/chinese_roberta_L-4_H-256](https://huggingface.co/uer/chinese_roberta_L-4_H-256)，BERT 结构、4 层、隐层 256，固定到 `68d674689f3c9d425e5436149c8acece077187f1`。此前使用的 [UER Tiny](https://huggingface.co/uer/chinese_roberta_L-2_H-128) 为 2 层、隐层 128，固定到 `374ab7ade2513a8ecfeada0348ae8bb59d52ce1c`。两者使用 RoBERTa 风格预训练目标，不是标准 12 层 BERT-Base。
- 数据：[PaddleNLP 提供的 MSRA NER](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/msra_ner.py)。脚本下载 `train.tsv` 与 `test.tsv` 并校验 MD5；文件仅保存在本机 `.cache/`。
- 学习参考：[NER-Pytorch-Chinese](https://github.com/xinyi-code/NER-Pytorch-Chinese) 的模型组合。本仓库的训练、CRF 解码和网页代码独立实现。

## 安装与运行（Windows PowerShell）

安装 Python 3.11 后，在本仓库目录运行。以下 512 句、2 轮命令只用于快速检查流程；要复现下方成绩，请使用完整实验命令。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py --model mini
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --model mini --train-size 512 --dev-size 128 --test-size 200 --epochs 2
.\.venv\Scripts\python.exe bert_bilstm_crf.py predict --text "张三在北京大学学习，后来去了上海。"
.\.venv\Scripts\python.exe demo_server.py
```

浏览器打开 `http://127.0.0.1:8766/`。以后也可双击 `start_demo.bat`，保持命令窗口开启。Mini 预训练权重约 35 MB；首次训练需要联网下载 MSRA 数据。完整 CPU 实验可能需要较长时间。要扩大实验，可提高 `--train-size`、`--dev-size`、`--test-size` 和 `--epochs`。

网页可输入最多 2000 字，并优先按句号、问号、感叹号等标点把长段落切成模型能处理的片段；空格与换行保留在原文中，结果的字符位置相对于完整原文。训练和测试仍使用不超过 96 字的独立句子，因此下方 F1 **不代表**长段落的实测准确率。若实体刚好跨越分段边界，识别结果可能不完整；演示时建议使用有正常标点的段落。命令行 `predict` 仍只接受模型设置的单句长度。

网页提供 8 条可选示例，涵盖校园、科研合作、高校论坛、跨城出差等场景。给导师展示时还可点击“加载进阶案例”，识别三段包含重复人名、机构名和地名的文本，查看类别统计与位置，再导出 JSON。推荐讲解顺序见 [进阶 Demo 指南](DEMO_GUIDE.md)。

## 已完成的本机实验

当前 demo 使用 UER Mini：先在 40000 条训练句和 1024 条开发句上训练 5 轮，再从该权重以较低学习率继续训练 1 轮。新检查点的开发集 F1 为 **93.68%**，高于原检查点的 **93.43%**；再尝试 1 轮后降至 93.29%，因此保留前一轮。在同一批**此前未用过**的 500 条测试句上，原检查点 F1 **89.55%**，当前 Demo 权重 F1 **90.42%**，提高 **0.87 个百分点**。详细计数、逐类结果和训练记录见 [实验报告](bert_bilstm_crf_report.md)。继续训练只加载模型权重，优化器会重新初始化。演示输入“张三在北京大学学习，后来去了上海。”时，模型识别出张三（人名）、北京大学（机构名）、上海（地名）。

此前分别训练过 2048 句、8192 句和 40000 句 Tiny 模型；原 5 轮 Mini 的测试行排除了这三次已使用的测试行。在**当时同一批测试句**上，40000 句 Tiny 的 F1 为 **86.8%**，原 Mini 为 **91.1%**。本次新测试集又排除了原 Mini 已使用的测试行。**91.1% 与 90.42% 来自两批不同测试句，不能直接比较。**各实验的配置、抽样行号及逐类指标在 `experiments/`；模型权重不上传。重新训练原模型，再继续训练 1 轮可用：

```powershell
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --model mini --train-size 40000 --dev-size 1024 --test-size 500 --epochs 5 --batch-size 16 --threads 8 --seed 2029 --exclude-test-lines experiments\pilot_run.json --exclude-test-lines experiments\final_run.json --exclude-test-lines experiments\large_run.json
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --model mini --train-size 40000 --dev-size 1024 --test-size 500 --epochs 1 --batch-size 16 --threads 8 --seed 2029 --bert-lr 0.00001 --head-lr 0.0003 --init-checkpoint .cache\bert_bilstm_crf\best.pt --output-dir .cache\mini_resume --report .cache\mini_resume_report.md --exclude-test-lines experiments\pilot_run.json --exclude-test-lines experiments\final_run.json --exclude-test-lines experiments\large_run.json --exclude-test-lines experiments\mini_run.json
```

继续训练时必须保持训练和开发样本行号相同；`--init-checkpoint` 会校验这些行号，训练中每轮写入 `progress.json`，中断后可用 `--init-run` 指向该文件恢复权重。以上命令把候选权重保存在单独目录，评估后再决定是否切换 Demo。`--device auto` 默认优先使用可用的 CUDA，否则使用 CPU。此次本机运行使用 CPU。

## 评估规则

训练和开发样本只从 `train.tsv` 抽取，测试样本只从 `test.tsv` 抽取；种子默认 2026，样本原始行号写入 `.cache/bert_bilstm_crf/run.json`。每轮在开发集计算实体级 F1，保存开发集最优 checkpoint；每次训练只在开发集模型选择完成后用该 checkpoint 评估对应测试抽样。实体类型及字符起止位置完全一致才计为正确。

结果写入 `bert_bilstm_crf_report.md`。小样本结果只能说明本机流程跑通，不代表完整数据集的论文复现分数。若需要和已发表工作比较，应保持训练集规模、模型、超参数和评估协议一致。

## 文件

- `data.py`：公开数据下载与 MD5 校验。
- `download_model.py`：下载固定版本的 UER Tiny 或 Mini 预训练模型。
- `bert_bilstm_crf.py`：字符与 token 对齐、模型、训练、评估和预测。
- `demo_server.py`、`demo.html`：本地网页演示。
- `DEMO_GUIDE.md`：进阶案例的展示顺序和结果解释。
- `test_bert_bilstm_crf.py`：CRF 精确性和数据划分测试。
- `compare_checkpoints.py`：在同一测试抽样上比较两个已冻结的 checkpoint。
- `experiments/`：各次实验的配置、抽样行号、同集对照结果及本机软件版本，方便核对和复现。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_bert_bilstm_crf.py
```

代码使用 MIT 许可；数据和预训练模型的来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

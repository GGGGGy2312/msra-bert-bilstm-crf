# MSRA 中文 NER：BERT-BiLSTM-CRF

一个**独立**的中文命名实体识别学习仓库。用公开 MSRA NER 数据微调 UER 中文 BERT 结构轻量模型、双向 LSTM 与线性链 CRF，识别人名、机构名、地名。仓库不依赖 LTP，也不包含数据集或预训练权重。

## 模型与数据

```text
输入字符 → UER Chinese RoBERTa-Tiny（BERT 结构）→ BiLSTM → 线性层 → CRF → BIO 标签 → 实体跨度
```

- 预训练模型：[uer/chinese_roberta_L-2_H-128](https://huggingface.co/uer/chinese_roberta_L-2_H-128)，BERT 结构、2 层、隐层 128，固定到 `374ab7ade2513a8ecfeada0348ae8bb59d52ce1c`。它使用 RoBERTa 风格预训练目标，不是标准 12 层 BERT-Base。
- 数据：[PaddleNLP 提供的 MSRA NER](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/msra_ner.py)。脚本下载 `train.tsv` 与 `test.tsv` 并校验 MD5；文件仅保存在本机 `.cache/`。
- 学习参考：[NER-Pytorch-Chinese](https://github.com/xinyi-code/NER-Pytorch-Chinese) 的模型组合。本仓库的训练、CRF 解码和网页代码独立实现。

## 安装与运行（Windows PowerShell）

安装 Python 3.11 后，在本仓库目录运行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_model.py
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --train-size 512 --dev-size 128 --test-size 200 --epochs 2
.\.venv\Scripts\python.exe bert_bilstm_crf.py predict --text "张三在北京大学学习，后来去了上海。"
.\.venv\Scripts\python.exe demo_server.py
```

浏览器打开 `http://127.0.0.1:8766/`。以后也可双击 `start_demo.bat`，保持命令窗口开启。首次模型权重下载约 13 MB；首次训练需要联网下载 MSRA 数据。CPU 训练仍可能需要几分钟。要扩大实验，可提高 `--train-size`、`--dev-size`、`--test-size` 和 `--epochs`。

## 已完成的本机实验

在 8192/1024/500 句训练/开发/测试抽样、6 轮训练下，本机测试集实体级 Precision **82.6%**、Recall **79.2%**、F1 **80.9%**。这只是小样本学习实验，不能作为完整 MSRA 数据集的论文复现成绩。详细类别结果和训练曲线见 [实验报告](bert_bilstm_crf_report.md)。演示输入“张三在北京大学学习，后来去了上海。”时，模型识别出张三（人名）、北京大学（机构名）、上海（地名）。

先做过一次 2048 句训练的探索实验；正式实验的 500 句测试样本排除了那次已看过的测试行。两次实验的配置、抽样行号及逐类指标保存在 `experiments/pilot_run.json` 和 `experiments/final_run.json`，模型权重不上传。重新训练最终实验可用：

```powershell
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --train-size 8192 --dev-size 1024 --test-size 500 --epochs 6 --seed 2027 --exclude-test-lines experiments\pilot_run.json
```

## 评估规则

训练和开发样本只从 `train.tsv` 抽取，测试样本只从 `test.tsv` 抽取；种子默认 2026，样本原始行号写入 `.cache/bert_bilstm_crf/run.json`。每轮在开发集计算实体级 F1，保存开发集最优 checkpoint；每次训练只在开发集模型选择完成后用该 checkpoint 评估对应测试抽样。实体类型及字符起止位置完全一致才计为正确。

结果写入 `bert_bilstm_crf_report.md`。小样本结果只能说明本机流程跑通，不代表完整数据集的论文复现分数。若需要和已发表工作比较，应保持训练集规模、模型、超参数和评估协议一致。

## 文件

- `data.py`：公开数据下载与 MD5 校验。
- `download_model.py`：下载固定版本的 UER 中文 BERT 结构预训练模型。
- `bert_bilstm_crf.py`：字符与 token 对齐、模型、训练、评估和预测。
- `demo_server.py`、`demo.html`：本地网页演示。
- `test_bert_bilstm_crf.py`：CRF 精确性和数据划分测试。
- `experiments/`：已完成两次实验的配置、抽样行号及结果，方便核对和复现。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_bert_bilstm_crf.py
```

代码使用 MIT 许可；数据和预训练模型的来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

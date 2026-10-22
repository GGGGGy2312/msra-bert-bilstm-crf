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

网页可输入最多 2000 字，并优先按句号、问号、感叹号等标点把长段落切成模型能处理的片段；空格与换行保留在原文中，结果的字符位置相对于完整原文。对不得不在词中间切开的片段，Demo 会额外读取切点前后的上下文，尝试恢复跨界实体。训练和测试仍使用不超过 96 字的独立句子，因此下方 F1 **不代表**长段落的实测准确率。边界恢复并不能保证识别正确；演示时建议使用有正常标点的段落。切点实验见 [边界补充推理报告](BOUNDARY_STUDY.md)。命令行 `predict` 仍只接受模型设置的单句长度。

网页提供 11 条可选示例，选择后自动载入，涵盖校园、科研合作、高校论坛、跨城出差和两条较长的中文材料；其中“中文项目记录（更长）”有三段、341 字。给导师展示时还可点击“载入多段进阶案例”，识别三段包含重复人名、机构名和地名的文本。结果支持按类别筛选、点击高亮词定位表格行、点击表格行定位原文，再复制或下载完整 JSON。输入框支持 Ctrl + Enter 开始识别。推荐讲解顺序见 [进阶 Demo 指南](DEMO_GUIDE.md)。

本机保留上一版权重时，识别结果下方还能点击“对比上一版模型”，在**同一段文本**上查看两版模型共同与独有的实体。新增的“开发集差异案例（同文对照）”来自公开 MSRA 开发样本，仅用于演示预测差异；它不是独立测试。克隆仓库后若没有上一版权重，对照按钮会隐藏，普通识别仍可使用。

## 已完成的本机实验

**当前 Demo** 在此前 UER Mini 权重插值检查点上，以小学习率用原 40000 句训练行中的 10000 句继续训练 1 轮。开发集实体级 F1 **93.96% → 94.17%**。在未用于当前训练或开发的 500 句 `train.tsv` 留出样本上，同集 F1 **93.90% → 94.45%**；在先前已经使用过的固定 500 句 `test.tsv` 抽样上补充核验，同集 F1 **90.31% → 90.91%**。后者是复用测试抽样，不能当作新的盲测成绩。详细结果和复现命令见 [小学习率继续训练实验](LOW_LR_STUDY.md)。

上一版权重插值方法：旧 5 轮检查点占 25%，继续训练检查点占 75%；1024 句开发集 F1 为 93.96%，当时新抽取的 500 句测试抽样 F1 为 90.31%。见 [权重插值实验](BLEND_STUDY.md)。

此前单独继续训练曾使另一批 500 句上的 F1 从 **89.55%** 提高到 **90.42%**。**90.42% 与本轮 90.31% 来自不同测试句，不可直接比较。**历史训练及同集对照见 [实验报告](bert_bilstm_crf_report.md)。

此前试过双检查点预测分数融合：开发集 F1 达 **94.18%**，但在另选的 154 句测试抽样上与当时的单模型同为 **93.50%**，因此没有采用。详情见 [分数融合试验](ENSEMBLE_STUDY.md)；它与本轮小学习率训练是两种不同方法。

此前分别训练过 2048 句、8192 句和 40000 句 Tiny 模型；原 5 轮 Mini 的测试行排除了这三次已使用的测试行。在**当时同一批测试句**上，40000 句 Tiny 的 F1 为 **86.8%**，原 Mini 为 **91.1%**。后来权重插值实验的新测试抽样排除了此前用过的 2500 条测试行；本轮继续训练复用了该抽样并另设未训练行留出集。各实验配置、抽样行号和逐类指标在 `experiments/`；模型权重不上传。重新训练原模型，再继续训练 1 轮可用：

```powershell
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --model mini --train-size 40000 --dev-size 1024 --test-size 500 --epochs 5 --batch-size 16 --threads 8 --seed 2029 --output-dir .cache\mini_previous --exclude-test-lines experiments\pilot_run.json --exclude-test-lines experiments\final_run.json --exclude-test-lines experiments\large_run.json
.\.venv\Scripts\python.exe bert_bilstm_crf.py train --model mini --train-size 40000 --dev-size 1024 --test-size 500 --epochs 1 --batch-size 16 --threads 8 --seed 2029 --bert-lr 0.00001 --head-lr 0.0003 --init-checkpoint .cache\mini_previous\best.pt --output-dir .cache\mini_resume --report .cache\mini_resume_report.md --exclude-test-lines experiments\pilot_run.json --exclude-test-lines experiments\final_run.json --exclude-test-lines experiments\large_run.json --exclude-test-lines experiments\mini_run.json
```

通常继续训练须保持训练与开发行号相同；本轮使用 `--resume-subset-size`，允许从原训练行中固定抽取子集，同时严格保持开发行号相同，并设置 `--test-size 0`。训练中每轮写入 `progress.json`，中断后可用 `--init-run` 指向该文件恢复权重。原始和插值权重的重建步骤见 [权重插值实验](BLEND_STUDY.md)，本轮命令见 [小学习率继续训练实验](LOW_LR_STUDY.md)。`--device auto` 默认优先使用可用的 CUDA，否则使用 CPU；此次本机运行使用 CPU。

## 评估规则

训练和开发样本从 `train.tsv` 抽取。每轮在开发集计算实体级 F1，保存开发集最优 checkpoint；实体类型及字符起止位置完全一致才计为正确。本轮额外从 `train.tsv` 未用于当前训练和开发的行中留出 500 句评估，并在既有 `test.tsv` 抽样上做复用对照。所有抽样行号记录在 `experiments/`，数据来源和是否曾使用在报告中分别标明。

结果写入 `bert_bilstm_crf_report.md`。小样本结果只能说明本机流程跑通，不代表完整数据集的论文复现分数。若需要和已发表工作比较，应保持训练集规模、模型、超参数和评估协议一致。

## 文件

- `data.py`：公开数据下载与 MD5 校验。
- `download_model.py`：下载固定版本的 UER Tiny 或 Mini 预训练模型。
- `bert_bilstm_crf.py`：字符与 token 对齐、模型、训练、评估和预测。
- `demo_server.py`、`demo.html`：本地网页演示。
- `DEMO_GUIDE.md`：进阶案例的展示顺序和结果解释。
- `test_bert_bilstm_crf.py`：CRF 精确性和数据划分测试。
- `compare_checkpoints.py`：在同一测试抽样上比较两个已冻结的 checkpoint。
- `blend_checkpoints.py`：开发集选择检查点插值比例，复核已发布的独立测试对照。
- `evaluate_logit_ensemble.py`：复核未采用的双检查点预测分数融合试验。
- `evaluate_low_lr_candidate.py`、`activate_low_lr_candidate.py`：复核本轮训练并在两项同集对照提高后激活权重。
- `evaluate_boundary_recovery.py`：用开发集构造跨切点长段落，对比旧分段与边界补充推理。
- `experiments/`：各次实验的配置、抽样行号、同集对照结果及本机软件版本，方便核对和复现。

运行测试：

```powershell
.\.venv\Scripts\python.exe -m unittest -v test_bert_bilstm_crf.py
```

代码使用 MIT 许可；数据和预训练模型的来源见 [THIRD_PARTY.md](THIRD_PARTY.md)。

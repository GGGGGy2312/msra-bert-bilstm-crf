# Data and pretrained model

This repository contains original experiment code under the MIT license. It
does not contain the MSRA corpus or pretrained BERT weights.

- MSRA NER is fetched from PaddleNLP's published dataset archive by `data.py`.
  The script verifies the archive and extracted TSV checksums listed in
  [PaddleNLP's loader](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/datasets/msra_ner.py).
  Consult the dataset provider for corpus usage terms.
- `uer/chinese_roberta_L-2_H-128` and `uer/chinese_roberta_L-4_H-256` are
  downloaded from their pinned [Tiny](https://huggingface.co/uer/chinese_roberta_L-2_H-128)
  and [Mini](https://huggingface.co/uer/chinese_roberta_L-4_H-256) model
  repositories. They use BERT architecture with RoBERTa-style pretraining. The UER-py
  [source repository](https://github.com/dbiir/UER-py) uses Apache-2.0;
  the model card does not state a separate license for its weights.
- The model combination is a common architecture also implemented in
  [NER-Pytorch-Chinese](https://github.com/xinyi-code/NER-Pytorch-Chinese).
  This repository does not copy that project's implementation.


# All Information Extraction tasks are Machine Reading Comprehensible

Integrated from [Learn-REx](https://github.com/Spico197/Learn-REx)

This project provides a comprehensive example for illustrating the functionalities of [REx](https://github.com/Spico197/REx).

Currently, this repo contains a Named Entity Recognition task via Machine Reading Comprehension strategy.

## 🌴 Dependencies

- Basics
  - torch
  - pytorch-rex==0.1.8
  - transformers

- Recommended (for dev as in `Makefile`)
  - formatting tools: isort, black
  - linting: flake8
  - testing: pytest, coverage

## 💾 Data Preprocessing

Download the dataset via `python data/download.py`.

## 🚀 QuickStart

1. download dataset
2. check `data/formatted/role2query.json` file
3. change configurations in `custom.yaml`
4. run `bash run.sh` to start training
5. change `skip_train` in `outputs/bert_mrc_ner/task_params.yaml` to `true`
6. try `inference.py` to make predictions
7. try debugging via VSCode debugger (the launch file locates in `.vscode/launch.json`)

## ✉️ Contact

If you found any problems or you have ideas to improve this project,
feel free to contact me via [GitHub](https://github.com/Spico197/REx) issues.


## 🗜️ TODO

- [ ] masked `multilabel_categorical_crossentropy`
- [ ] retain negative samples when transforming evaluation sets so that evaluations are accurate no matter what negative probs are
- [ ] hyper-param tuning, lr, dropout
- [ ] full negative samples

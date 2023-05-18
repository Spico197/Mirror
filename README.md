# All Information Extraction tasks are Machine Reading Comprehensible

## 🌴 Dependencies

- Basics
  - torch
  - pytorch-rex==0.10b0 (github latest commit)
  - transformers
  - pandas
  - openpyxl
  - rich

- Recommended (for dev as in `Makefile`)
  - formatting tools: isort, black
  - linting: flake8
  - testing: pytest, coverage

## 💾 Data Preprocessing

Check bash files in `resources/Mirror/v1.3`.

## 🚀 QuickStart

- training

```bash
nohup bash run.sh ddp 1>logs/InstructBert_TagSpan_DebertaV3Base_MergedPretrainedData.log 2>&1 &
```


## 🗜️ TODO

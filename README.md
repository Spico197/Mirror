<!-- ---
title: Mirror
emoji: 🪞
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
---
 -->

<div align="center">
  <h1>🪞 Mirror: A Universal Framework for Various Information Extraction Tasks</h1>
  <img src="figs/mirror-framework.png" alt="Mirror Framework">
</div>

<hr>

## 🔥 Supported Tasks

1. Named Entity Recognition
2. Entity Relationship Extraction (Triplet Extraction)
3. Event Extraction
4. Aspect-based Sentiment Analysis
5. Multi-span Extraction (e.g. Discontinuous NER)
6. N-ary Extraction (e.g. Hyper Relation Extraction)
7. Extractive Machine Reading Comprehension (MRC) and Question Answering
8. Classification & Multi-choice MRC

![System Comparison](figs/sys-comparison.png)

## 🌴 Dependencies

Python>=3.10

```bash
pip install -r requirements.txt
```

## 🚀 QuickStart

### Pretrained Model Weights & Datasets

Download the pretrained model weights & datasets from [[Anonymized OSF]](https://osf.io/kwsm4/?view_only=91a610f7a81a430eb953378f26a8054c) .

No worries, it's an anonymous link just for double blind peer reviewing.

### Pretraining

1. Download and unzip the pretraining corpus into `resources/Mirror/v1.4_sampled_v3/merged/all_excluded`
2. Start to run

```bash
CUDA_VISIBLE_DEVICES=0 rex train -m src.task -dc conf/Pretrain_excluded.yaml
```

### Fine-tuning

⚠️ Due to data license constraints, some datasets are unavailable to provide directly (e.g. ACE04, ACE05).

1. Download and unzip the pretraining corpus into `resources/Mirror/v1.4_sampled_v3/merged/all_excluded`
2. Download and unzip the fine-tuning datasets into `resources/Mirror/uie/`
3. Start to fine-tuning

```bash
# UIE tasks
CUDA_VISIBLE_DEVICES=0 bash scripts/single_task_wPTAllExcluded_wInstruction/run1.sh
CUDA_VISIBLE_DEVICES=1 bash scripts/single_task_wPTAllExcluded_wInstruction/run2.sh
CUDA_VISIBLE_DEVICES=2 bash scripts/single_task_wPTAllExcluded_wInstruction/run3.sh
CUDA_VISIBLE_DEVICES=3 bash scripts/single_task_wPTAllExcluded_wInstruction/run4.sh
# Multi-span and N-ary extraction
CUDA_VISIBLE_DEVICES=4 bash scripts/single_task_wPTAllExcluded_wInstruction/run_new_tasks.sh
# GLUE datasets
CUDA_VISIBLE_DEVICES=5 bash scripts/single_task_wPTAllExcluded_wInstruction/glue.sh
```

### Analysis Experiments

- Few-shot experiments : `scripts/run_fewshot.sh`. Collecting results: `python mirror_fewshot_outputs/get_avg_results.py`
- Mirror w/ PT w/o Inst. : `scripts/single_task_wPTAllExcluded_woInstruction`
- Mirror w/o PT w/ Inst. : `scripts/single_task_wo_pretrain`
- Mirror w/o PT w/o Inst. : `scripts/single_task_wo_pretrain_wo_instruction`

### Evaluation

1. Change `task_dir` and `data_pairs` you want to evaluate. The default setting is to get results of Mirror<sub>direct</sub> on all downstream tasks.
2. `CUDA_VISIBLE_DEVICES=0 python -m src.eval`

### Demo

1. Download and unzip the pretrained task dump into `mirror_outputs/Mirror_Pretrain_AllExcluded_2`
2. Try our demo:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.app.api_backend
```

![Demo](figs/mirror-demo.gif)

## 💌 Others

This project is licensed under Apache-2.0.
We hope you enjoy it ~

<hr>
<div align="center">
  <p>Mirror Team w/ 💖</p>
</div>

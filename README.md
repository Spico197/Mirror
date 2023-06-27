# All for One and One for All: A Multi-Task Framework for Various Information Extraction Tasks

## 🌴 Dependencies

Python>=3.10

```bash
pip install -r requirements.txt
```

## 🗃️ Preparation

- Tasks
  - `mirror_outputs/Mirror_Pretrain_AllExcluded_2`
- Data
  - `resources/Mirror/uie`
  - `resources/Mirror/v1.4_uie_fewshot`

## 📑 TODO

- training:
  - 546 GB disk space consumption in total
  - `CUDA_VISIBLE_DEVICES="2" nohup rex train ... 1>logs/xxx.log 2>&1 &`


- collecting results: `python mirror_fewshot_outputs/get_avg_results.py`
- paste the results to latex project directly

# All for One and One for All: A Multi-Task Framework for Various Information Extraction Tasks

## 🌴 Dependencies

Python>=3.10

```bash
pip install -r requirements.txt
```

## 🗃️ Preparation

- Tasks
  - `mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap`
  - `mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap`
- Data
  - `resources/Mirror/uie`

## 📑 TODO

```bash
# included pretrain
CUDA_VISIBLE_DEVICES=0 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_IncludedPretrain_MultiTask"
# excluded pretrain
CUDA_VISIBLE_DEVICES=1 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_MultiTask"
# excluded pretrain single task
CUDA_VISIBLE_DEVICES=2 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -c conf/uie_data/ent_conll03.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_Ent_CoNLL03"
CUDA_VISIBLE_DEVICES=3 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -c conf/uie_data/rel_conll04.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_Rel_CoNLL04"
CUDA_VISIBLE_DEVICES=4 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -c conf/uie_data/event_ace05.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_Event_ACE05"
CUDA_VISIBLE_DEVICES=5 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -c conf/uie_data/absa_16res.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_ABSA_16res"
# ablation: without pretrain
CUDA_VISIBLE_DEVICES=6 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -a "base_model_path=null" "task_name=Mirror_woPT_MultiTask"
# ablation: without pretrain & instruction
CUDA_VISIBLE_DEVICES=7 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data_woInstruction.yaml -a "base_model_path=null" "task_name=Mirror_woPT_woInstruction_MultiTask"
```

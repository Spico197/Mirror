#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"


pretrain() {
    # rex train -m src.task -dc conf/Pretrain_woOverlapV2.yaml
    rex train -m src.task -dc conf/Pretrain_v1.5.yaml
    rex train -m src.task -dc conf/Pretrain_excluded.yaml
    rex train -m src.task -dc conf/Pretrain_v1.5_woInstruction.yaml
}

single() {
    # python wait.py \
    #     --task_name="InstructBert_TagW2_DebertaV3Base_ACE05EN_NerRelEvent" \
    #     --cuda="4" \
    #     --wait="schedule" \
    #     --req_gpu_num=1

    # # uie multi-task without pretraining
    # rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_uie_data.yaml

    # uie multi-task fine-tuning with pretrain
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_uie_data.yaml -a train_filepath=/data/tzhu/Mirror/resources/Mirror/uie/merged/train_wo_overlap_v2.jsonl base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth task_name=Mirror_UIE_wPT_woOverlapV2

        # -c conf/uie_data/rel_scierc.yaml #  \
        # -c conf/merge_uie_data.yaml
        # -dc conf/mirror-ace05en.yaml \
        # -a task_name=InstructBert_TagW2_DebertaV3Base_ACE05EN_NerRelEvent \
        #     label_span=tag mode=w2

    # rex train -m src.task -dc conf/ner.yaml
    # rex train -m src.task -dc conf/mrc.yaml
}

ddp() {
    # GPU_SCOPE="0,1,2,3"
    # REQ_GPU_NUM=2
    # GPUS="1"
    # GPUS=$(python wait.py --task_name="$TASK_NAME" --cuda=$GPU_SCOPE --wait="schedule" --req_gpu_num=$REQ_GPU_NUM)
    # torchrun --nnodes=1 --nproc_per_node=2 -m rex.cmds.train -m src.task -dc conf/mirror-multi-task-pretrain.yaml
    # torchrun --nnodes=1 --nproc_per_node=2 -m rex.cmds.train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/t-rex_pretrain.yaml
    CUDA_VISIBLE_DEVICES="2,3" torchrun --nnodes=1 --nproc_per_node=2 -m rex.cmds.train -m src.task -dc conf/Pretrain_v1.5.yaml
}

uie_wo_pretrain() {
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml -a base_model_path=null
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml -a base_model_path=null
}

uie() {
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml
}

new_tasks() {
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/cadec.yaml
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/hyperred.yaml
}

pretraining_analysis() {
    # # included pretrain
    # rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_IncludedPretrain_MultiTask"
    # excluded pretrain
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_MultiTask"
    # # excluded pretrain single task
    # rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -c conf/uie_data/ent_conll03.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_Ent_CoNLL03"
    # rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -c conf/uie_data/rel_conll04.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_Rel_CoNLL04"
    # rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -c conf/uie_data/event_ace05.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_Event_ACE05"
    # rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -c conf/uie_data/absa_16res.yaml -a "base_model_path=mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap/ckpt/SchemaGuidedInstructBertModel.best.pth" "task_name=Mirror_ExcludedPretrain_ABSA_16res"
}

ablation() {
    # without pretrain
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data.yaml -a "base_model_path=null" "task_name=Mirror_woPT_MultiTask"
    # without pretrain & instruction
    rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/merge_analysis_data_woInstruction.yaml -a "base_model_path=null" "task_name=Mirror_woPT_woInstruction_MultiTask"
}


RUN_METHOD=$1
case $RUN_METHOD in
    ddp)
        ddp
        ;;
    uie)
        uie
        ;;
    new_tasks)
        new_tasks
        ;;
    pretraining_analysis)
        pretraining_analysis
        ;;
    ablation)
        ablation
        ;;
    *)
        single
        ;;
esac

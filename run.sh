#!/bin/bash

export CUDA_VISIBLE_DEVICES="3"


single() {
    # python wait.py \
    #     --task_name="InstructBert_TagW2_DebertaV3Base_ACE05EN_NerRelEvent" \
    #     --cuda="4" \
    #     --wait="schedule" \
    #     --req_gpu_num=1

    rex train \
        -m src.task \
        -dc conf/mirror-multi-task-pretrain.yaml # \
        # -c conf/merge_uie_data.yaml
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
    torchrun --nnodes=1 --nproc_per_node=2 -m rex.cmds.train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/t-rex_pretrain.yaml
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


RUN_METHOD=$1
case $RUN_METHOD in
    ddp)
        ddp
        ;;
    uie)
        uie
        ;;
    *)
        single
        ;;
esac

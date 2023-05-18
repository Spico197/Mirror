#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"


single() {
    # python wait.py \
    #     --task_name="InstructBert_TagW2_DebertaV3Base_ACE05EN_NerRelEvent" \
    #     --cuda="4" \
    #     --wait="schedule" \
    #     --req_gpu_num=1

    rex train \
        -m src.task \
        -dc conf/mirror-ace05en.yaml #  \
        # -dc conf/mirror-multi-task-pretrain.yaml #  \
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


RUN_METHOD=$1
case $RUN_METHOD in
    ddp)
        ddp
        ;;
    *)
        single
        ;;
esac

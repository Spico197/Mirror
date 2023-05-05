#!/bin/bash

export CUDA_VISIBLE_DEVICES=2


single() {
    # python wait.py \
    #     --task_name="InstructBert_TagW2_DebertaV3Base_ACE05EN_NerRelEvent" \
    #     --cuda="4" \
    #     --wait="schedule" \
    #     --req_gpu_num=1

    rex train \
        -m src.task \
        -dc conf/mirror-ace05en.yaml #  \
        # -a task_name=InstructBert_TagW2_DebertaV3Base_ACE05EN_NerRelEvent \
        #     label_span=tag mode=w2

    # rex train -m src.task -dc conf/ner.yaml
    # rex train -m src.task -dc conf/mrc.yaml
}

ddp() {
    torchrun --nnodes=1 --nproc_per_node=4 -m rex.cmds.train -m src.task -dc conf/custom.yaml
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

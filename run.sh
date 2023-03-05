#!/bin/bash

export CUDA_VISIBLE_DEVICES=2


single() {
    rex train -m src.task -dc conf/custom.yaml
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

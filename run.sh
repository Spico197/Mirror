#!/bin/bash

export CUDA_VISIBLE_DEVICES=3


single() {
    rex train -m src -dc custom.yaml
}

ddp() {
    torchrun --nnodes=1 --nproc_per_node=4 -m rex.cmds.train -m src -dc custom.yaml
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

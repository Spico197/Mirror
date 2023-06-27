export CUDA_VISIBLE_DEVICES="3"

accelerate launch \
    --config_file conf/ac/g1_dpspd.yaml \
    -m rex.cmds.train \
        -m src.task \
        -dc conf/mirror-multi-task-pretrain.yaml

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/cadec.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_DisNER_CADEC
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/hyperred.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_HyperRel_HyperRED

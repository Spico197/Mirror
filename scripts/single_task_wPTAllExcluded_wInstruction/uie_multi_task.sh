CUDA_VISIBLE_DEVICES=3 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/merged.yaml -a task_name=Mirror_MultiTask_UIE_wPT_wInst

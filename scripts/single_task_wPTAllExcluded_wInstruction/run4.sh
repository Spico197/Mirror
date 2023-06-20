# GPUS=$(python wait.py --task_name="wPT_wInst_run4" --cuda="0,1,2,3" --wait="schedule" --req_gpu_num=1)
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_16res.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_ABSA_16res
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14lap.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_ABSA_14lap
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_15res.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_ABSA_15res

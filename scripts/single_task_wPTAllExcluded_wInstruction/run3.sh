rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Event_ACE05
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Event_CASIE
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_ABSA_14res

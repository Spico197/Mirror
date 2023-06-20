# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Event_ACE05 data_dir=resources/Mirror/uie/event/ace05-evt/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Event_CASIE data_dir=resources/Mirror/uie/event/casie/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_ABSA_14res data_dir=resources/Mirror/uie/absa/14res/remove_instruction

# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a base_model_path=null data_dir=resources/Mirror/uie/event/ace05-evt/remove_instruction task_name=Mirror_SingleTask_woPT_woInstruction_Event_ACE05
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a base_model_path=null data_dir=resources/Mirror/uie/event/casie/remove_instruction task_name=Mirror_SingleTask_woPT_woInstruction_Event_CASIE
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a base_model_path=null data_dir=resources/Mirror/uie/absa/14res/remove_instruction task_name=Mirror_SingleTask_woPT_woInstruction_Ent_ABSA_14res

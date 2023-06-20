# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Ent_CoNLL03 data_dir=resources/Mirror/uie/ent/conll03/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Ent_ACE04 data_dir=resources/Mirror/uie/ent/ace04/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Ent_ACE05 data_dir=resources/Mirror/uie/ent/ace05/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Rel_ACE05 resources/Mirror/uie/rel/ace05-rel/remove_instruction

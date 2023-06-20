# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Rel_CoNLL04 data_dir=resources/Mirror/uie/rel/conll04/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Rel_SciERC data_dir=resources/Mirror/uie/rel/scierc/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_woInst_Rel_NYT data_dir=resources/Mirror/uie/rel/nyt/remove_instruction

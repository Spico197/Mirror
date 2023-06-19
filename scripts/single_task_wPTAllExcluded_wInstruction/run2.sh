# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Rel_CoNLL04
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Rel_SciERC
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Rel_NYT

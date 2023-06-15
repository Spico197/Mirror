# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace04.yaml -a base_model_path=null task_name=Mirror_SingleTask_woPT_Ent_ACE04
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_ace05.yaml -a base_model_path=null task_name=Mirror_SingleTask_woPT_Ent_ACE05
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/ent_conll03.yaml -a base_model_path=null task_name=Mirror_SingleTask_woPT_Ent_CoNLL03
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_ace05.yaml -a base_model_path=null task_name=Mirror_SingleTask_woPT_Rel_ACE05

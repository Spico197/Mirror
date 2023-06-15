# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a base_model_path=null data_dir=resources/Mirror/uie/rel/conll04/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a base_model_path=null data_dir=resources/Mirror/uie/rel/nyt/remove_instruction
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a base_model_path=null data_dir=resources/Mirror/uie/rel/scierc/remove_instruction

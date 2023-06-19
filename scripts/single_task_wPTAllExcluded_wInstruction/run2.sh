# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_conll04.yaml -a base_model_path=mirror_outputs/Mirror_Pretrain_AllExcluded_2/ckpt/SchemaGuidedInstructBertModel.step.139999.pth task_name=Mirror_SingleTask_wPTAllExcludedStep139999_Rel_CoNLL04
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_nyt.yaml -a base_model_path=mirror_outputs/Mirror_Pretrain_AllExcluded_2/ckpt/SchemaGuidedInstructBertModel.step.139999.pth task_name=Mirror_SingleTask_wPTAllExcludedStep139999_Rel_NYT
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/rel_scierc.yaml -a base_model_path=mirror_outputs/Mirror_Pretrain_AllExcluded_2/ckpt/SchemaGuidedInstructBertModel.step.139999.pth task_name=Mirror_SingleTask_wPTAllExcludedStep139999_Rel_SciERC

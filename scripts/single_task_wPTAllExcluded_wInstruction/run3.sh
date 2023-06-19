# uie single task w/o pretrain

rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_ace05.yaml -a base_model_path=mirror_outputs/Mirror_Pretrain_AllExcluded_2/ckpt/SchemaGuidedInstructBertModel.step.139999.pth task_name=Mirror_SingleTask_wPTAllExcludedStep139999_Event_ACE05
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/event_casie.yaml -a base_model_path=mirror_outputs/Mirror_Pretrain_AllExcluded_2/ckpt/SchemaGuidedInstructBertModel.step.139999.pth task_name=Mirror_SingleTask_wPTAllExcludedStep139999_Event_CASIE
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/absa_14res.yaml -a base_model_path=mirror_outputs/Mirror_Pretrain_AllExcluded_2/ckpt/SchemaGuidedInstructBertModel.step.139999.pth task_name=Mirror_SingleTask_wPTAllExcludedStep139999_ABSA_14res

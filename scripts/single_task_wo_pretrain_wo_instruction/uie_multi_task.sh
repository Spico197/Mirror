CUDA_VISIBLE_DEVICES=6 rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/merged.yaml -a task_name=Mirror_MultiTask_UIE_woPT_woInst base_model_path=null include_instructions=false

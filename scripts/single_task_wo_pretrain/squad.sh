CUDA_VISIBLE_DEVICES=3 nohup rex train \
    -m src.task \
    -dc conf/mirror-multi-task-pretrain.yaml \
    -c conf/nlu/plm.yaml \
    -c conf/nlu/squad_v2.yaml \
    -a base_model_path=null task_name=Mirror_SingleTask_woPT_Span_SQuADv2 dropout=0.1 learning_rate=1e-5 other_learning_rate=1e-5 num_epochs=10 plm_dir=/data/tzhu/PLM/microsoft--deberta-v3-large \
    1>logs/Mirror_SingleTask_woPT_Span_SQuADv2.log 2>&1 &

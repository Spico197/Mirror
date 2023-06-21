rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/rte.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Cls_RTE
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/qnli.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Cls_QNLI
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/mrpc.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Cls_MRPC
rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/nlu/plm.yaml -c conf/nlu/squad_v2.yaml -a task_name=Mirror_SingleTask_wPTAllExcluded_Span_SQuADv2

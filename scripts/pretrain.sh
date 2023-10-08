CUDA_VISIBLE_DEVICES=0 nohup rex train -m src.task -dc conf/Pretrain_excluded.yaml -a task_name=Mirror_Pretrain_AllExcluded_2_woCls plm_dir=/data/tzhu/PLM/microsoft--deberta-v3-large data_dir=resources/Mirror/v1.4_sampled_v3/merged/wo_cls 1>logs/pretrain_woCls.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup rex train -m src.task -dc conf/Pretrain_excluded.yaml -a task_name=Mirror_Pretrain_AllExcluded_2_woIe plm_dir=/data/tzhu/PLM/microsoft--deberta-v3-large data_dir=resources/Mirror/v1.4_sampled_v3/merged/wo_ie 1>logs/pretrain_woIe.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup rex train -m src.task -dc conf/Pretrain_excluded.yaml -a task_name=Mirror_Pretrain_AllExcluded_2_woSpan plm_dir=/data/tzhu/PLM/microsoft--deberta-v3-large data_dir=resources/Mirror/v1.4_sampled_v3/merged/wo_span 1>logs/pretrain_woSpan.log 2>&1 &

# job 2: 3500, 3: 3553, 4: 3609

#!/bin/bash

export CUDA_VISIBLE_DEVICES="2"

# 546 GB disk space consumption in total
# # CUDA_VISIBLE_DEVICES="2" nohup rex train ... 1>logs/xxx.log 2>&1 &

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed1_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed1/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed1_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed1/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed1_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed1/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed1_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed1/1shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed1_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed1/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed1_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed1/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed1_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed1/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed1_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed1/5shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed1_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed1/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed1_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed1/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed1_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed1/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed1_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed1/10shot"


# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed2_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed2/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed2_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed2/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed2_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed2/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed2_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed2/1shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed2_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed2/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed2_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed2/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed2_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed2/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed2_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed2/5shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed2_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed2/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed2_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed2/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed2_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed2/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed2_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed2/10shot"


# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed3_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed3/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed3_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed3/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed3_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed3/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed3_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed3/1shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed3_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed3/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed3_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed3/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed3_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed3/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed3_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed3/5shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed3_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed3/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed3_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed3/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed3_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed3/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed3_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed3/10shot"


# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed4_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed4/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed4_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed4/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed4_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed4/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed4_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed4/1shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed4_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed4/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed4_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed4/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed4_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed4/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed4_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed4/5shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed4_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed4/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed4_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed4/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed4_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed4/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed4_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed4/10shot"


# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed5_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed5/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed5_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed5/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed5_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed5/1shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed5_1shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed5/1shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed5_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed5/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed5_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed5/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed5_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed5/5shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed5_5shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed5/5shot"

# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed5_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed5/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed5_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed5/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed5_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed5/10shot"
# rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed5_10shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed5/10shot"


uie_fewshot() {
    for seed in $(seq 1 5); do
        for shot in {1,5,10}; do
            echo "seed=$seed, shot=$shot"
            echo "=== ent_conll03 ==="
            rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/ent_conll03.yaml -a "task_name=Mirror_SingleTask_Ent_CoNLL03_seed${seed}_${shot}shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/ent/conll03/seed${seed}/${shot}shot"
            echo "=== rel_conll04 ==="
            rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/rel_conll04.yaml -a "task_name=Mirror_SingleTask_Rel_CoNLL04_seed${seed}_${shot}shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/rel/conll04/seed${seed}/${shot}shot"
            echo "=== event_ace05 ==="
            rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/event_ace05.yaml -a "task_name=Mirror_SingleTask_Event_ACE05_seed${seed}_${shot}shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/event/ace05-evt/seed${seed}/${shot}shot"
            echo "=== ent_ace04 ==="
            rex train -m src.task -dc conf/mirror-multi-task-pretrain.yaml -c conf/uie_data/wPretrain.yaml -c conf/uie_data/fewshot.yaml -c conf/uie_data/absa_16res.yaml -a "task_name=Mirror_SingleTask_ABSA_16res_seed${seed}_${shot}shot" "data_dir=resources/Mirror/v1.4_uie_fewshot/absa/16res/seed${seed}/${shot}shot"
        done
    done
}

uie_fewshot

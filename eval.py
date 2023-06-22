from pathlib import Path

import pandas as pd
from rex.utils.initialization import set_seed_and_log_path
from rex.utils.io import load_json
from rich.console import Console
from rich.table import Table

from src.task import SchemaGuidedInstructBertTask

set_seed_and_log_path(log_path="eval.log")


# task_dir = "mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData"
# task_dir = "mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2"
# task_dir = "mirror_outputs/InstructBert_NewMergedUIEData"
# task_dir = "mirror_outputs/InstructBert_Large_NewMergedUIEData"
# task_dir = "mirror_outputs/InstructBert_Large_NewMergedUIEData_bs10"
# task_dir = "outputs/InstructBert_TagSpan_DebertaV3Base_ACE05ENPlus"
# task_dir = "mirror_outputs/MirrorLarge_SamplingPretrain"
# task_dir = "mirror_outputs/Mirror_UIE_wPT"
# task_dir = "mirror_outputs/Mirror_UIE_wPT_woOverlapV2"
# task_dir = "mirror_outputs/Mirror_ExcludedPretrain_MultiTask"
# task_dir = "mirror_outputs/MirrorLarge_SamplingPretrain_woZeroShotNER"
# task_dir = "mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap"
# task_dir = "mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap"
# task_dir = "mirror_outputs/Mirror_Pretrain_DataV1.5_2"
task_dir = "mirror_outputs/Mirror_Pretrain_AllExcluded_2"
# task_dir = "mirror_outputs/Mirror_Pretrain_DataV1.5_woInstruction"
# task_dir = "mirror_outputs/Mirror_SingleTask_wPTAllExcluded_Rel_NYT"
# task_dir = "mirror_outputs/Mirror_SingleTask_wPTAllExcluded_Rel_CoNLL04"
# task_dir = "mirror_outputs/Mirror_SingleTask_wPTAllExcluded_woInst_Rel_NYT"
task: SchemaGuidedInstructBertTask = SchemaGuidedInstructBertTask.from_taskdir(
    task_dir,
    load_best_model=True,
    initialize=False,
    dump_configfile=False,
    update_config={
        "regenerate_cache": True,
        "eval_on_data": ["dev"],
        "select_best_on_data": "dev",
        "select_best_by_key": "metric",
        "best_metric_field": "general_spans.micro.f1",
        "eval_batch_size": 32,
    },
)
table = Table(title=task_dir)

data_pairs = [
    # fmt: off

    # # UIE eval data
    ["ent_ace04_test", "resources/Mirror/uie/ent/ace04/test.jsonl"],
    ["ent_ace05_test", "resources/Mirror/uie/ent/ace05/test.jsonl"],
    ["ent_conll03_test", "resources/Mirror/uie/ent/conll03/test.jsonl"],
    ["rel_ace05_test", "resources/Mirror/uie/rel/ace05-rel/test.jsonl"],
    ["rel_conll04_test", "resources/Mirror/uie/rel/conll04/test.jsonl"],
    ["rel_nyt_test", "resources/Mirror/uie/rel/nyt/test.jsonl"],
    ["rel_scierc_test", "resources/Mirror/uie/rel/scierc/test.jsonl"],
    ["event_ace05_test", "resources/Mirror/uie/event/ace05-evt/test.jsonl"],
    ["event_casie_test", "resources/Mirror/uie/event/casie/test.jsonl"],
    ["absa_14res_test", "resources/Mirror/uie/absa/14res/test.jsonl"],
    ["absa_14lap_test", "resources/Mirror/uie/absa/14lap/test.jsonl"],
    ["absa_15res_test", "resources/Mirror/uie/absa/15res/test.jsonl"],
    ["absa_16res_test", "resources/Mirror/uie/absa/16res/test.jsonl"],
    # # discontinuous NER
    ["discontinuous_ent", "resources/Mirror/new_abilities_v2/cadec/new/test.jsonl"],
    # # hyper-RE
    ["hyper_rel", "resources/Mirror/new_abilities_v2/HyperRED/new/test.jsonl"],
    # # analysis
    # ["ent_conll03_test", "resources/Mirror/uie/ent/conll03/test.jsonl"],
    # ["rel_conll04_test", "resources/Mirror/uie/rel/conll04/test.jsonl"],
    # ["event_ace05_test", "resources/Mirror/uie/event/ace05-evt/test.jsonl"],
    # ["absa_16res_test", "resources/Mirror/uie/absa/16res/test.jsonl"],
    # # zero-shot NER
    ["ent_movie", "resources/Mirror/v1.3/ent/en/MIT_MOVIE_Review/instructed/test.jsonl"],
    ["ent_restaurant", "resources/Mirror/v1.3/ent/en/MIT_Restaurant_Review/instructed/test.jsonl"],
    ["ent_ai", "resources/Mirror/v1.3/ent/en/CrossNER_AI/instructed/test.jsonl"],
    ["ent_literature", "resources/Mirror/v1.3/ent/en/CrossNER_literature/instructed/test.jsonl"],
    ["ent_music", "resources/Mirror/v1.3/ent/en/CrossNER_music/instructed/test.jsonl"],
    ["ent_politics", "resources/Mirror/v1.3/ent/en/CrossNER_politics/instructed/test.jsonl"],
    ["ent_science", "resources/Mirror/v1.3/ent/en/CrossNER_science/instructed/test.jsonl"],
    # # zero-shot NER w/o instructions
    # ["ent_movie", "resources/Mirror/v1.4/ent/en/MIT_MOVIE_Review/instructed/remove_instruction/test.jsonl"],
    # ["ent_restaurant", "resources/Mirror/v1.4/ent/en/MIT_Restaurant_Review/instructed/remove_instruction/test.jsonl"],
    # ["ent_ai", "resources/Mirror/v1.4/ent/en/CrossNER_AI/instructed/remove_instruction/test.jsonl"],
    # ["ent_literature", "resources/Mirror/v1.4/ent/en/CrossNER_literature/instructed/remove_instruction/test.jsonl"],
    # ["ent_music", "resources/Mirror/v1.4/ent/en/CrossNER_music/instructed/remove_instruction/test.jsonl"],
    # ["ent_politics", "resources/Mirror/v1.4/ent/en/CrossNER_politics/instructed/remove_instruction/test.jsonl"],
    # ["ent_science", "resources/Mirror/v1.4/ent/en/CrossNER_science/instructed/remove_instruction/test.jsonl"],
    # # cls
    # ["cls_agnews", "resources/Mirror/v1.4/cls/en/ag_news/instructed/test.jsonl"],
    # # NER RandomICL
    # ["ent_MIT_MOVIE_Review_ICL", "resources/Mirror/ner_icl/MIT_MOVIE_Review.jsonl"],
    # ["ent_MIT_Restaurant_Review_ICL", "resources/Mirror/ner_icl/MIT_Restaurant_Review.jsonl"],
    # ["ent_CrossNER_AI_ICL", "resources/Mirror/ner_icl/CrossNER_AI.jsonl"],
    # ["ent_CrossNER_literature_ICL", "resources/Mirror/ner_icl/CrossNER_literature.jsonl"],
    # ["ent_CrossNER_music_ICL", "resources/Mirror/ner_icl/CrossNER_music.jsonl"],
    # ["ent_CrossNER_politics_ICL", "resources/Mirror/ner_icl/CrossNER_politics.jsonl"],
    # ["ent_CrossNER_science_ICL", "resources/Mirror/ner_icl/CrossNER_science.jsonl"],
    # NER Retrieval
    # ["ent_MIT_MOVIE_Review_Retrieval", "resources/Mirror/ner_web_enhanced_bg/MIT_MOVIE_Review.jsonl"],
    # ["ent_MIT_Restaurant_Review_Retrieval", "resources/Mirror/ner_web_enhanced_bg/MIT_Restaurant_Review.jsonl"],
    # ["ent_CrossNER_AI_Retrieval", "resources/Mirror/ner_web_enhanced_bg/CrossNER_AI.jsonl"],
    # ["ent_CrossNER_literature_Retrieval", "resources/Mirror/ner_web_enhanced_bg/CrossNER_literature.jsonl"],
    # ["ent_CrossNER_music_Retrieval", "resources/Mirror/ner_web_enhanced_bg/CrossNER_music.jsonl"],
    # ["ent_CrossNER_politics_Retrieval", "resources/Mirror/ner_web_enhanced_bg/CrossNER_politics.jsonl"],
    # ["ent_CrossNER_science_Retrieval", "resources/Mirror/ner_web_enhanced_bg/CrossNER_science.jsonl"],
    # mrc
    ["span_squad2", "resources/Mirror/v1.3/span/en/squad_v2/dev.jsonl"],
    # # glue
    ["cls_glue_cola", "resources/Mirror/v1.3/cls/en/CoLA/formated/dev.jsonl"],
    ["cls_glue_qqp", "resources/Mirror/v1.3/cls/en/QQP/new/dev.jsonl"],
    ["cls_glue_mnli", "resources/Mirror/v1.3/cls/en/MNLI/formated/MNLI_dev.jsonl"],
    ["cls_glue_sst2", "resources/Mirror/v1.3/cls/en/SST-2/instructed/SST-2_dev.jsonl"],
    ["cls_glue_qnli", "resources/Mirror/v1.3/cls/en/QNLI/processed/QNLI_dev.jsonl"],
    ["cls_glue_rte", "resources/Mirror/v1.3/cls/en/RTE/formated/RTE_dev.jsonl"],
    ["cls_glue_mrpc", "resources/Mirror/v1.3/cls/en/MRPC/formated/dev.jsonl"],
    # Mirror v1.4 all train
    # ["cls_ag_news_train", "resources/Mirror/v1.4/cls/en/ag_news/instructed/train.jsonl"],
    # ["cls_ANLI_R1_train", "resources/Mirror/v1.4/cls/en/ANLI/R1_processed/train.jsonl"],
    # ["cls_ANLI_R2_train", "resources/Mirror/v1.4/cls/en/ANLI/R2_processed/train.jsonl"],
    # ["cls_ANLI_R3_train", "resources/Mirror/v1.4/cls/en/ANLI/R3_processed/train.jsonl"],
    # ["cls_ARC_train", "resources/Mirror/v1.4/cls/en/ARC/new/train.jsonl"],
    # ["cls_CoLA_train", "resources/Mirror/v1.4/cls/en/CoLA/formated/train.jsonl"],
    # ["cls_CosmosQA_train", "resources/Mirror/v1.4/cls/en/CosmosQA/new/train.jsonl"],
    # ["cls_dbpedia_train", "resources/Mirror/v1.4/cls/en/dbpedia/new/train.jsonl"],
    # ["cls_DREAM_train", "resources/Mirror/v1.4/cls/en/DREAM/new/train.jsonl"],
    # ["cls_MedQA_train", "resources/Mirror/v1.4/cls/en/MedQA/new/train.jsonl"],
    # ["cls_MRPC_train", "resources/Mirror/v1.4/cls/en/MRPC/formated/train.jsonl"],
    # ["cls_OpenBookQA_train", "resources/Mirror/v1.4/cls/en/OpenBookQA/new/train.jsonl"],
    # ["cls_RACE_train", "resources/Mirror/v1.4/cls/en/RACE/instructed/train.jsonl"],
    # ["cls_RACE-_trainC", "resources/Mirror/v1.4/cls/en/RACE-C/new/train.jsonl"],
    # ["cls_SciQ_train", "resources/Mirror/v1.4/cls/en/SciQ/instructed/train.jsonl"],
    # ["cls_SNLI_train", "resources/Mirror/v1.4/cls/en/SNLI/instructed/train.jsonl"],
    # ["ent_ace04_train", "resources/Mirror/v1.4/ent/en/ace04/train.jsonl"],
    # ["ent_ace05-uie_train", "resources/Mirror/v1.4/ent/en/ace05-uie/train.jsonl"],
    # ["ent_AnatEM_train", "resources/Mirror/v1.4/ent/en/AnatEM/instructed/train.jsonl"],
    # ["ent_bc2gm_train", "resources/Mirror/v1.4/ent/en/bc2gm/instructed/train.jsonl"],
    # ["ent_bc4chemd_train", "resources/Mirror/v1.4/ent/en/bc4chemd/instructed/train.jsonl"],
    # ["ent_bc5cdr_train", "resources/Mirror/v1.4/ent/en/bc5cdr/instructed/train.jsonl"],
    # ["ent_Broad_Tweet_Corpus_train", "resources/Mirror/v1.4/ent/en/Broad_Tweet_Corpus/instructed/train.jsonl"],
    # ["ent_conll03_train", "resources/Mirror/v1.4/ent/en/conll03/train.jsonl"],
    # ["ent_CrossNER_AI_train", "resources/Mirror/v1.4/ent/en/CrossNER_AI/instructed/train.jsonl"],
    # ["ent_CrossNER_literature_train", "resources/Mirror/v1.4/ent/en/CrossNER_literature/instructed/train.jsonl"],
    # ["ent_CrossNER_music_train", "resources/Mirror/v1.4/ent/en/CrossNER_music/instructed/train.jsonl"],
    # ["ent_CrossNER_politics_train", "resources/Mirror/v1.4/ent/en/CrossNER_politics/instructed/train.jsonl"],
    # ["ent_CrossNER_science_train", "resources/Mirror/v1.4/ent/en/CrossNER_science/instructed/train.jsonl"],
    # ["ent_FabNER_train", "resources/Mirror/v1.4/ent/en/FabNER/instructed/train.jsonl"],
    # ["ent_FindVehicle_train", "resources/Mirror/v1.4/ent/en/FindVehicle/instructed/train.jsonl"],
    # ["ent_GENIA_NER_train", "resources/Mirror/v1.4/ent/en/GENIA_NER/instructed/train.jsonl"],
    # ["ent_HarveyNER_train", "resources/Mirror/v1.4/ent/en/HarveyNER/instructed/train.jsonl"],
    # ["ent_MIT_MOVIE_Review_train", "resources/Mirror/v1.4/ent/en/MIT_MOVIE_Review/instructed/train.jsonl"],
    # ["ent_MIT_Restaurant_Review_train", "resources/Mirror/v1.4/ent/en/MIT_Restaurant_Review/instructed/train.jsonl"],
    # ["ent_MultiNERD_train", "resources/Mirror/v1.4/ent/en/MultiNERD/instructed/train.jsonl"],
    # ["ent_NCBIdiease_train", "resources/Mirror/v1.4/ent/en/NCBIdiease/instructed/train.jsonl"],
    # ["ent_ontoNotes5_train", "resources/Mirror/v1.4/ent/en/ontoNotes5/instructed/train.jsonl"],
    # ["ent_TweetNER7_train", "resources/Mirror/v1.4/ent/en/TweetNER7/instructed/train.jsonl"],
    # ["ent_WikiANN_en_train", "resources/Mirror/v1.4/ent/en/WikiANN_en/instructed/train.jsonl"],
    # ["ent_WNUT-16_train", "resources/Mirror/v1.4/ent/en/WNUT-16/train.jsonl"],
    # ["event_ace05-evt-uie_train", "resources/Mirror/v1.4/event/en/ace05-evt-uie/train.jsonl"],
    # ["event_casie_train", "resources/Mirror/v1.4/event/en/casie/train.jsonl"],
    # ["event_PHEE_train", "resources/Mirror/v1.4/event/en/PHEE/instructed/train.jsonl"],
    # ["rel_14lap_train", "resources/Mirror/v1.4/rel/en/14lap/train.jsonl"],
    # ["rel_14res_train", "resources/Mirror/v1.4/rel/en/14res/train.jsonl"],
    # ["rel_15res_train", "resources/Mirror/v1.4/rel/en/15res/train.jsonl"],
    # ["rel_16res_train", "resources/Mirror/v1.4/rel/en/16res/train.jsonl"],
    # ["rel_ace05-rel-uie_train", "resources/Mirror/v1.4/rel/en/ace05-rel-uie/train.jsonl"],
    # ["rel_conll04_train", "resources/Mirror/v1.4/rel/en/conll04/train.jsonl"],
    # ["rel_nyt_multi_train", "resources/Mirror/v1.4/rel/en/nyt_multi/train.jsonl"],
    # ["rel_scierc_train", "resources/Mirror/v1.4/rel/en/scierc/train.jsonl"],
    # # ["span_BiPaR_train", "resources/Mirror/v1.4/span/en/BiPaR/train.jsonl"],  # x
    # ["span_SubjQA_books_train", "resources/Mirror/v1.4/span/en/SubjQA/books/train.jsonl"],
    # ["span_SubjQA_electronics_train", "resources/Mirror/v1.4/span/en/SubjQA/electronics/train.jsonl"],
    # ["span_SubjQA_grocery_train", "resources/Mirror/v1.4/span/en/SubjQA/grocery/train.jsonl"],
    # ["span_SubjQA_movies_train", "resources/Mirror/v1.4/span/en/SubjQA/movies/train.jsonl"],
    # ["span_SubjQA_restaurants_train", "resources/Mirror/v1.4/span/en/SubjQA/restaurants/train.jsonl"],
    # ["span_SubjQA_tripadvisor_train", "resources/Mirror/v1.4/span/en/SubjQA/tripadvisor/train.jsonl"],
    # ["span_ms_marco_v2.1", "resources/Mirror/v1.4/span/en/ms_marco_v2.1/train.jsonl"],
    # ["span_newsqa", "resources/Mirror/v1.4/span/en/newsqa/train.jsonl"],
    # ["span_squad_v2", "resources/Mirror/v1.4/span/en/squad_v2/train.jsonl"],
    # # Mirror v1.4 all test
    # ["cls_ag_news_test", "resources/Mirror/v1.4/cls/en/ag_news/instructed/test.jsonl"],
    # ["cls_ANLI_R1_test", "resources/Mirror/v1.4/cls/en/ANLI/R1_processed/test.jsonl"],
    # ["cls_ANLI_R2_test", "resources/Mirror/v1.4/cls/en/ANLI/R2_processed/test.jsonl"],
    # ["cls_ANLI_R3_test", "resources/Mirror/v1.4/cls/en/ANLI/R3_processed/test.jsonl"],
    # ["cls_ARC_test", "resources/Mirror/v1.4/cls/en/ARC/new/test.jsonl"],
    # ["cls_CoLA_test", "resources/Mirror/v1.4/cls/en/CoLA/formated/test.jsonl"],
    # ["cls_CosmosQA_test", "resources/Mirror/v1.4/cls/en/CosmosQA/new/test.jsonl"],
    # ["cls_dbpedia_test", "resources/Mirror/v1.4/cls/en/dbpedia/new/test.jsonl"],
    # ["cls_DREAM_test", "resources/Mirror/v1.4/cls/en/DREAM/new/test.jsonl"],
    # ["cls_MedQA_test", "resources/Mirror/v1.4/cls/en/MedQA/new/test.jsonl"],
    # ["cls_MRPC_test", "resources/Mirror/v1.4/cls/en/MRPC/formated/test.jsonl"],
    # ["cls_OpenBookQA_test", "resources/Mirror/v1.4/cls/en/OpenBookQA/new/test.jsonl"],
    # ["cls_RACE_test", "resources/Mirror/v1.4/cls/en/RACE/instructed/test.jsonl"],
    # ["cls_RACE-C_test", "resources/Mirror/v1.4/cls/en/RACE-C/new/test.jsonl"],
    # ["cls_SciQ_test", "resources/Mirror/v1.4/cls/en/SciQ/instructed/test.jsonl"],
    # ["cls_SNLI_test", "resources/Mirror/v1.4/cls/en/SNLI/instructed/test.jsonl"],
    # ["ent_ace04_test", "resources/Mirror/v1.4/ent/en/ace04/test.jsonl"],
    # ["ent_ace05-uie_test", "resources/Mirror/v1.4/ent/en/ace05-uie/test.jsonl"],
    # ["ent_AnatEM_test", "resources/Mirror/v1.4/ent/en/AnatEM/instructed/test.jsonl"],
    # ["ent_bc2gm_test", "resources/Mirror/v1.4/ent/en/bc2gm/instructed/test.jsonl"],
    # ["ent_bc4chemd_test", "resources/Mirror/v1.4/ent/en/bc4chemd/instructed/test.jsonl"],
    # ["ent_bc5cdr_test", "resources/Mirror/v1.4/ent/en/bc5cdr/instructed/test.jsonl"],
    # ["ent_Broad_Tweet_Corpus_test", "resources/Mirror/v1.4/ent/en/Broad_Tweet_Corpus/instructed/test.jsonl"],
    # ["ent_conll03_test", "resources/Mirror/v1.4/ent/en/conll03/test.jsonl"],
    # ["ent_CrossNER_AI_test", "resources/Mirror/v1.4/ent/en/CrossNER_AI/instructed/test.jsonl"],
    # ["ent_CrossNER_literature_test", "resources/Mirror/v1.4/ent/en/CrossNER_literature/instructed/test.jsonl"],
    # ["ent_CrossNER_music_test", "resources/Mirror/v1.4/ent/en/CrossNER_music/instructed/test.jsonl"],
    # ["ent_CrossNER_politics_test", "resources/Mirror/v1.4/ent/en/CrossNER_politics/instructed/test.jsonl"],
    # ["ent_CrossNER_science_test", "resources/Mirror/v1.4/ent/en/CrossNER_science/instructed/test.jsonl"],
    # ["ent_FabNER_test", "resources/Mirror/v1.4/ent/en/FabNER/instructed/test.jsonl"],
    # ["ent_FindVehicle_test", "resources/Mirror/v1.4/ent/en/FindVehicle/instructed/test.jsonl"],
    # ["ent_GENIA_NER_test", "resources/Mirror/v1.4/ent/en/GENIA_NER/instructed/test.jsonl"],
    # ["ent_HarveyNER_test", "resources/Mirror/v1.4/ent/en/HarveyNER/instructed/test.jsonl"],
    # ["ent_MIT_MOVIE_Review_test", "resources/Mirror/v1.4/ent/en/MIT_MOVIE_Review/instructed/test.jsonl"],
    # ["ent_MIT_Restaurant_Review_test", "resources/Mirror/v1.4/ent/en/MIT_Restaurant_Review/instructed/test.jsonl"],
    # ["ent_MultiNERD_test", "resources/Mirror/v1.4/ent/en/MultiNERD/instructed/test.jsonl"],
    # ["ent_NCBIdiease_test", "resources/Mirror/v1.4/ent/en/NCBIdiease/instructed/test.jsonl"],
    # ["ent_ontoNotes5_test", "resources/Mirror/v1.4/ent/en/ontoNotes5/instructed/test.jsonl"],
    # ["ent_TweetNER7_test", "resources/Mirror/v1.4/ent/en/TweetNER7/instructed/test.jsonl"],
    # ["ent_WikiANN_en_test", "resources/Mirror/v1.4/ent/en/WikiANN_en/instructed/test.jsonl"],
    # ["ent_WNUT-16_test", "resources/Mirror/v1.4/ent/en/WNUT-16/test.jsonl"],
    # ["event_ace05-evt-uie_test", "resources/Mirror/v1.4/event/en/ace05-evt-uie/test.jsonl"],
    # ["event_casie_test", "resources/Mirror/v1.4/event/en/casie/test.jsonl"],
    # ["event_PHEE_test", "resources/Mirror/v1.4/event/en/PHEE/instructed/test.jsonl"],
    # ["rel_14lap_test", "resources/Mirror/v1.4/rel/en/14lap/test.jsonl"],
    # ["rel_14res_test", "resources/Mirror/v1.4/rel/en/14res/test.jsonl"],
    # ["rel_15res_test", "resources/Mirror/v1.4/rel/en/15res/test.jsonl"],
    # ["rel_16res_test", "resources/Mirror/v1.4/rel/en/16res/test.jsonl"],
    # ["rel_ace05-rel-uie_test", "resources/Mirror/v1.4/rel/en/ace05-rel-uie/test.jsonl"],
    # ["rel_conll04_test", "resources/Mirror/v1.4/rel/en/conll04/test.jsonl"],
    # ["rel_nyt_multi_test", "resources/Mirror/v1.4/rel/en/nyt_multi/test.jsonl"],
    # ["rel_scierc_test", "resources/Mirror/v1.4/rel/en/scierc/test.jsonl"],
    # ["span_BiPaR_test", "resources/Mirror/v1.4/span/en/BiPaR/test.jsonl"],
    # ["span_SubjQA_books_test", "resources/Mirror/v1.4/span/en/SubjQA/books/test.jsonl"],
    # ["span_SubjQA_electronics_test", "resources/Mirror/v1.4/span/en/SubjQA/electronics/test.jsonl"],
    # ["span_SubjQA_grocery_test", "resources/Mirror/v1.4/span/en/SubjQA/grocery/test.jsonl"],
    # ["span_SubjQA_movies_test", "resources/Mirror/v1.4/span/en/SubjQA/movies/test.jsonl"],
    # ["span_SubjQA_restaurants_test", "resources/Mirror/v1.4/span/en/SubjQA/restaurants/test.jsonl"],
    # ["span_SubjQA_tripadvisor_test", "resources/Mirror/v1.4/span/en/SubjQA/tripadvisor/test.jsonl"],
    # fmt: on
]

eval_res = {"task": [], "dataset": [], "metric_val": []}
table.add_column("Task", justify="left", style="cyan")
table.add_column("Dataset", justify="left", style="magenta")
table.add_column("Metric (%)", justify="right", style="green")
for dname, fpath in data_pairs:
    dname = dname.lower()
    task.data_manager.update_datapath(dname, fpath)
    _, res = task.eval(dname, verbose=True, dump=True, dump_middle=True)
    # res = load_json(Path(task_dir) / "measures" / f"{dname}.json")["metrics"]
    if dname.startswith("ent_"):
        eval_res["task"].append("ent")
        eval_res["dataset"].append(dname)
        eval_res["metric_val"].append(res["ent"]["micro"]["f1"])
    elif dname.startswith("rel_"):
        eval_res["task"].append("rel")
        eval_res["dataset"].append(dname)
        eval_res["metric_val"].append(res["rel"]["rel"]["micro"]["f1"])
    elif dname.startswith("event_"):
        eval_res["task"].append("event")
        eval_res["dataset"].append(dname + "_tgg")
        eval_res["metric_val"].append(res["event"]["trigger_cls"]["f1"])
        eval_res["task"].append("event")
        eval_res["dataset"].append(dname + "_arg")
        eval_res["metric_val"].append(res["event"]["arg_cls"]["f1"])
    elif dname.startswith("absa_"):
        eval_res["task"].append("absa")
        eval_res["dataset"].append(dname)
        eval_res["metric_val"].append(res["rel"]["rel"]["micro"]["f1"])
    elif dname.startswith("cls_"):
        eval_res["task"].append("cls")
        eval_res["dataset"].append(dname)
        if "_glue_" in dname:
            if "_cola" in dname:
                eval_res["metric_val"].append(res["cls"]["mcc"])
            else:
                eval_res["metric_val"].append(res["cls"]["acc"])
        else:
            eval_res["metric_val"].append(res["cls"]["mf1"]["micro"]["f1"])
    elif dname.startswith("span"):
        eval_res["task"].append("span_em")
        eval_res["dataset"].append(dname)
        eval_res["metric_val"].append(res["span"]["em"])
        eval_res["task"].append("span_f1")
        eval_res["dataset"].append(dname)
        eval_res["metric_val"].append(res["span"]["f1"]["f1"])
    elif dname.startswith("discontinuous_ent"):
        eval_res["task"].append("discontinuous_ent")
        eval_res["dataset"].append(dname)
        eval_res["metric_val"].append(res["discontinuous_ent"]["micro"]["f1"])
    elif dname.startswith("hyper_rel"):
        eval_res["task"].append("hyper_rel")
        eval_res["dataset"].append(dname)
        eval_res["metric_val"].append(res["hyper_rel"]["micro"]["f1"])
    else:
        raise ValueError

for i in range(len(eval_res["task"])):
    table.add_row(
        eval_res["task"][i],
        eval_res["dataset"][i],
        f"{100*eval_res['metric_val'][i]:.3f}",
    )

console = Console()
console.print(table)

df = pd.DataFrame(eval_res)
df.to_excel(task.measures_path.joinpath("data_eval_res.xlsx"))


"""
fixed upper bound

mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.934 │
│ ent       │ ent_ace05_test                 │         100.000 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │          96.444 │
│ rel       │ rel_conll04_test               │          96.009 │
│ rel       │ rel_nyt_test                   │          78.145 │
│ rel       │ rel_scierc_test                │          81.288 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │          92.987 │
│ event     │ event_casie_test_arg           │          93.376 │
│ absa      │ absa_14res_test                │          98.991 │
│ absa      │ absa_14lap_test                │          99.815 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.611 │
└───────────┴────────────────────────────────┴─────────────────┘

eval UIEData2
mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          84.912 │
│ ent       │ ent_ace05_test                 │          90.300 │
│ ent       │ ent_conll03_test               │          92.335 │
│ rel       │ rel_ace05_test                 │          59.879 │
│ rel       │ rel_conll04_test               │          45.272 │
│ rel       │ rel_nyt_test                   │          72.610 │
│ rel       │ rel_scierc_test                │          19.890 │
│ event     │ event_ace05_test_tgg           │          71.752 │
│ event     │ event_ace05_test_arg           │          51.140 │
│ event     │ event_casie_test_tgg           │          63.915 │
│ event     │ event_casie_test_arg           │          32.243 │
│ absa      │ absa_14res_test                │          75.456 │
│ absa      │ absa_14lap_test                │          64.251 │
│ absa      │ absa_15res_test                │          93.525 │
│ absa      │ absa_16res_test                │          76.505 │
└───────────┴────────────────────────────────┴─────────────────┘

eval UIEData2 upperbound fixed-v1 with constraint
mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.934 │
│ ent       │ ent_ace05_test                 │         100.000 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │          98.401 │
│ rel       │ rel_conll04_test               │          99.176 │
│ rel       │ rel_nyt_test                   │          78.573 │
│ rel       │ rel_scierc_test                │          89.655 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │          92.987 │
│ event     │ event_casie_test_arg           │          93.376 │
│ absa      │ absa_14res_test                │          99.091 │
│ absa      │ absa_14lap_test                │          99.815 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.708 │
└───────────┴────────────────────────────────┴─────────────────┘

eval UIEData2 fixed-v2 with constraint
mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          84.912 │
│ ent       │ ent_ace05_test                 │          90.300 │
│ ent       │ ent_conll03_test               │          92.335 │
│ rel       │ rel_ace05_test                 │          60.365 │
│ rel       │ rel_conll04_test               │          46.064 │
│ rel       │ rel_nyt_test                   │          73.048 │
│ rel       │ rel_scierc_test                │          20.084 │
│ event     │ event_ace05_test_tgg           │          71.752 │
│ event     │ event_ace05_test_arg           │          51.140 │
│ event     │ event_casie_test_tgg           │          63.915 │
│ event     │ event_casie_test_arg           │          32.243 │
│ absa      │ absa_14res_test                │          75.456 │
│ absa      │ absa_14lap_test                │          64.251 │
│ absa      │ absa_15res_test                │          93.525 │
│ absa      │ absa_16res_test                │          76.505 │
└───────────┴────────────────────────────────┴─────────────────┘

pso upper bound, find all nnw paths
mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData2
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.934 │
│ ent       │ ent_ace05_test                 │         100.000 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │          98.463 │
│ rel       │ rel_conll04_test               │          99.176 │
│ rel       │ rel_nyt_test                   │          98.392 │
│ rel       │ rel_scierc_test                │          92.593 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │          92.987 │
│ event     │ event_casie_test_arg           │          93.376 │
│ absa      │ absa_14res_test                │          98.947 │
│ absa      │ absa_14lap_test                │          99.815 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.709 │
└───────────┴────────────────────────────────┴─────────────────┘

MergedUIEDataMultitaskSFT
pso find all nnw paths
output label type to span len constraint
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.934 │
│ ent       │ ent_ace05_test                 │         100.000 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │         100.000 │
│ rel       │ rel_conll04_test               │          99.881 │
│ rel       │ rel_nyt_test                   │          99.362 │
│ rel       │ rel_scierc_test                │          97.113 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │          92.987 │
│ event     │ event_casie_test_arg           │          93.376 │
│ absa      │ absa_14res_test                │          99.496 │
│ absa      │ absa_14lap_test                │          99.908 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.903 │
└───────────┴────────────────────────────────┴─────────────────┘

pso upper bound
new merged uie data
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          99.951 │
│ ent       │ ent_ace05_test                 │          99.852 │
│ ent       │ ent_conll03_test               │         100.000 │
│ rel       │ rel_ace05_test                 │          99.957 │
│ rel       │ rel_conll04_test               │          99.643 │
│ rel       │ rel_nyt_test                   │          99.380 │
│ rel       │ rel_scierc_test                │          97.113 │
│ event     │ event_ace05_test_tgg           │         100.000 │
│ event     │ event_ace05_test_arg           │         100.000 │
│ event     │ event_casie_test_tgg           │         100.000 │
│ event     │ event_casie_test_arg           │          99.991 │
│ absa      │ absa_14res_test                │          99.496 │
│ absa      │ absa_14lap_test                │          99.908 │
│ absa      │ absa_15res_test                │          99.794 │
│ absa      │ absa_16res_test                │          99.903 │
└───────────┴────────────────────────────────┴─────────────────┘

merged uie data v2 eval on new data
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                        ┃      Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                 │          83.055 │
│ ent       │ ent_ace05_test                 │          89.497 │
│ ent       │ ent_conll03_test               │          86.333 │
│ rel       │ rel_ace05_test                 │          67.989 │
│ rel       │ rel_conll04_test               │           0.000 │
│ rel       │ rel_nyt_test                   │          91.656 │
│ rel       │ rel_scierc_test                │           7.509 │
│ event     │ event_ace05_test_tgg           │          71.170 │
│ event     │ event_ace05_test_arg           │          49.408 │
│ event     │ event_casie_test_tgg           │          30.459 │
│ event     │ event_casie_test_arg           │           7.966 │
│ absa      │ absa_14res_test                │          73.684 │
│ absa      │ absa_14lap_test                │          62.737 │
│ absa      │ absa_15res_test                │          90.928 │
│ absa      │ absa_16res_test                │          74.853 │
└───────────┴────────────────────────────────┴─────────────────┘

InstructBert_NewMergedUIEData middle
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task  ┃ Dataset             ┃ Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent   │ ent_ace04_test      │     85.686 │
│ ent   │ ent_ace05_test      │     90.354 │
│ ent   │ ent_conll03_test    │     92.456 │
│ rel   │ rel_ace05_test      │     61.343 │
│ rel   │ rel_conll04_test    │     67.797 │
│ rel   │ rel_nyt_test        │     92.122 │
│ rel   │ rel_scierc_test     │     21.911 │
│ event │ event_ace05_test_t… │     67.178 │
│ event │ event_ace05_test_a… │     43.394 │
│ event │ event_casie_test_t… │     59.827 │
│ event │ event_casie_test_a… │     37.390 │
│ absa  │ absa_14res_test     │     74.384 │
│ absa  │ absa_14lap_test     │     65.564 │
│ absa  │ absa_15res_test     │     85.775 │
│ absa  │ absa_16res_test     │     74.533 │
└───────┴─────────────────────┴────────────┘

mirror_outputs/InstructBert_NewMergedUIEData
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task  ┃ Dataset             ┃ Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent   │ ent_ace04_test      │     86.929 │
│ ent   │ ent_ace05_test      │     92.290 │
│ ent   │ ent_conll03_test    │     92.747 │
│ rel   │ rel_ace05_test      │     67.777 │
│ rel   │ rel_conll04_test    │     71.159 │
│ rel   │ rel_nyt_test        │     93.226 │
│ rel   │ rel_scierc_test     │     34.031 │
│ event │ event_ace05_test_t… │     72.372 │
│ event │ event_ace05_test_a… │     52.946 │
│ event │ event_casie_test_t… │     69.821 │
│ event │ event_casie_test_a… │     56.977 │
│ absa  │ absa_14res_test     │     75.732 │
│ absa  │ absa_14lap_test     │     66.401 │
│ absa  │ absa_15res_test     │     92.798 │
│ absa  │ absa_16res_test     │     74.138 │
└───────┴─────────────────────┴────────────┘

large model on new data
mirror_outputs/InstructBert_Large_NewMergedUIEData
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Task   ┃ Dataset                 ┃  Metric (%) ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ ent    │ ent_ace04_test          │      88.531 │
│ ent    │ ent_ace05_test          │      93.515 │
│ ent    │ ent_conll03_test        │      93.094 │
│ rel    │ rel_ace05_test          │      72.015 │
│ rel    │ rel_conll04_test        │      75.933 │
│ rel    │ rel_nyt_test            │      93.995 │
│ rel    │ rel_scierc_test         │      42.069 │
│ event  │ event_ace05_test_tgg    │      73.177 │
│ event  │ event_ace05_test_arg    │      57.833 │
│ event  │ event_casie_test_tgg    │      71.659 │
│ event  │ event_casie_test_arg    │      59.336 │
│ absa   │ absa_14res_test         │      76.899 │
│ absa   │ absa_14lap_test         │      63.448 │
│ absa   │ absa_15res_test         │      95.436 │
│ absa   │ absa_16res_test         │      75.624 │
└────────┴─────────────────────────┴─────────────┘

mirror_outputs/InstructBert_Large_NewMergedUIEData_bs10
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Task    ┃ Dataset                    ┃   Metric (%) ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ ent     │ ent_ace04_test             │       87.759 │
│ ent     │ ent_ace05_test             │       93.673 │
│ ent     │ ent_conll03_test           │       92.449 │
│ rel     │ rel_ace05_test             │       72.188 │
│ rel     │ rel_conll04_test           │       77.255 │
│ rel     │ rel_nyt_test               │       93.764 │
│ rel     │ rel_scierc_test            │       42.358 │
│ event   │ event_ace05_test_tgg       │       72.256 │
│ event   │ event_ace05_test_arg       │       58.561 │
│ event   │ event_casie_test_tgg       │       71.800 │
│ event   │ event_casie_test_arg       │       59.477 │
│ absa    │ absa_14res_test            │       77.663 │
│ absa    │ absa_14lap_test            │       66.142 │
│ absa    │ absa_15res_test            │       93.769 │
│ absa    │ absa_16res_test            │       74.835 │
└─────────┴────────────────────────────┴──────────────┘

pretrain direct infer
mirror_outputs/MirrorLarge_SamplingPretrain
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task  ┃ Dataset            ┃ Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent   │ ent_ace04_test     │     82.892 │
│ ent   │ ent_ace05_test     │     85.304 │
│ ent   │ ent_conll03_test   │     91.905 │
│ rel   │ rel_ace05_test     │      9.233 │
│ rel   │ rel_conll04_test   │     52.615 │
│ rel   │ rel_nyt_test       │     88.260 │
│ rel   │ rel_scierc_test    │      1.621 │
│ event │ event_ace05_test_… │     54.799 │
│ event │ event_ace05_test_… │     14.267 │
│ event │ event_casie_test_… │     19.541 │
│ event │ event_casie_test_… │      0.701 │
│ absa  │ absa_14res_test    │     59.305 │
│ absa  │ absa_14lap_test    │     57.208 │
│ absa  │ absa_15res_test    │     62.546 │
│ absa  │ absa_16res_test    │     67.333 │
└───────┴────────────────────┴────────────┘

pretrain direct infer
mirror_outputs/MirrorLarge_SamplingPretrain
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Task  ┃ Dataset           ┃  Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ ent   │ ent_movie         │      85.367 │
│ ent   │ ent_restaurant    │      81.790 │
│ ent   │ ent_ai            │      61.803 │
│ ent   │ ent_literature    │      66.210 │
│ ent   │ ent_music         │      75.516 │
│ ent   │ ent_politics      │      75.653 │
│ ent   │ ent_science       │      69.719 │
└───────┴───────────────────┴─────────────┘

pretrain w/o zero-shot NER, direct infer
mirror_outputs/MirrorLarge_SamplingPretrain_woZeroShotNER
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Task     ┃ Dataset                 ┃       Metric (%) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ ent      │ ent_movie               │           39.229 │
│ ent      │ ent_restaurant          │           22.413 │
│ ent      │ ent_ai                  │           51.155 │
│ ent      │ ent_literature          │           51.484 │
│ ent      │ ent_music               │           62.215 │
│ ent      │ ent_politics            │           62.087 │
│ ent      │ ent_science             │           52.632 │
└──────────┴─────────────────────────┴──────────────────┘

pretrain direct infer, upper-bound
mirror_outputs/MirrorLarge_SamplingPretrain
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task     ┃ Dataset         ┃ Metric (%) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cls      │ cls_glue_cola   │    100.000 │
│ cls      │ cls_glue_qqp    │    100.000 │
│ cls      │ cls_glue_mnli   │    100.000 │
│ cls      │ cls_glue_sst2   │    100.000 │
│ cls      │ cls_glue_qnli   │    100.000 │
│ cls      │ cls_glue_rte    │    100.000 │
│ cls      │ cls_glue_mrpc   │    100.000 │
│ span_em  │ span_squad2     │     95.614 │
│ span_f1  │ span_squad2     │     99.907 │
└──────────┴─────────────────┴────────────┘

pretrain direct infer on glue and mrc
mirror_outputs/MirrorLarge_SamplingPretrain
┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task     ┃ Dataset         ┃ Metric (%) ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cls      │ cls_glue_cola   │      0.000 │
│ cls      │ cls_glue_qqp    │     62.895 │
│ cls      │ cls_glue_mnli   │      0.000 │
│ cls      │ cls_glue_sst2   │     22.222 │
│ cls      │ cls_glue_qnli   │     43.053 │
│ cls      │ cls_glue_rte    │      0.000 │
│ cls      │ cls_glue_mrpc   │     68.382 │
│ span_em  │ span_squad2     │     38.664 │
│ span_f1  │ span_squad2     │     55.380 │
└──────────┴─────────────────┴────────────┘

mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Task    ┃ Dataset                  ┃   Metric (%) ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ ent     │ ent_ace04_test           │       84.426 │
│ ent     │ ent_ace05_test           │       86.357 │
│ ent     │ ent_conll03_test         │       92.716 │
│ rel     │ rel_ace05_test           │       22.812 │
│ rel     │ rel_conll04_test         │       53.652 │
│ rel     │ rel_nyt_test             │       89.094 │
│ rel     │ rel_scierc_test          │        7.832 │
│ event   │ event_ace05_test_tgg     │       63.624 │
│ event   │ event_ace05_test_arg     │       25.000 │
│ event   │ event_casie_test_tgg     │       50.017 │
│ event   │ event_casie_test_arg     │       17.642 │
│ absa    │ absa_14res_test          │       66.818 │
│ absa    │ absa_14lap_test          │       62.260 │
│ absa    │ absa_15res_test          │       62.896 │
│ absa    │ absa_16res_test          │       69.530 │
│ ent     │ ent_movie                │       85.942 │
│ ent     │ ent_restaurant           │       83.304 │
│ ent     │ ent_ai                   │       65.724 │
│ ent     │ ent_literature           │       67.932 │
│ ent     │ ent_music                │       78.245 │
│ ent     │ ent_politics             │       75.921 │
│ ent     │ ent_science              │       70.959 │
└─────────┴──────────────────────────┴──────────────┘

mirror_outputs/MirrorLarge_SamplingPretrain_woLowResource_woOverlap
┏━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ Task      ┃ Dataset                          ┃       Metric (%) ┃
┡━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ ent       │ ent_ace04_test                   │           84.325 │
│ ent       │ ent_ace05_test                   │           86.262 │
│ ent       │ ent_conll03_test                 │           69.106 │
│ rel       │ rel_ace05_test                   │           20.818 │
│ rel       │ rel_conll04_test                 │           16.601 │
│ rel       │ rel_nyt_test                     │           88.332 │
│ rel       │ rel_scierc_test                  │            3.910 │
│ event     │ event_ace05_test_tgg             │            0.000 │
│ event     │ event_ace05_test_arg             │            0.000 │
│ event     │ event_casie_test_tgg             │           39.003 │
│ event     │ event_casie_test_arg             │            9.116 │
│ absa      │ absa_14res_test                  │           63.170 │
│ absa      │ absa_14lap_test                  │           60.268 │
│ absa      │ absa_15res_test                  │           60.633 │
│ absa      │ absa_16res_test                  │           68.119 │
│ ent       │ ent_movie                        │           40.964 │
│ ent       │ ent_restaurant                   │           20.022 │
│ ent       │ ent_ai                           │           51.130 │
│ ent       │ ent_literature                   │           44.803 │
│ ent       │ ent_music                        │           60.626 │
│ ent       │ ent_politics                     │           61.190 │
│ ent       │ ent_science                      │           53.649 │
└───────────┴──────────────────────────────────┴──────────────────┘

mirror_outputs/Mirror_UIE_wPT
┏━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┓
┃       ┃         ┃  Metric ┃
┃ Task  ┃ Dataset ┃     (%) ┃
┡━━━━━━━╇━━━━━━━━━╇━━━━━━━━━┩
│ ent   │ ent_ac… │  85.526 │
│ ent   │ ent_ac… │  88.250 │
│ ent   │ ent_co… │  91.896 │
│ rel   │ rel_ac… │  40.157 │
│ rel   │ rel_co… │  60.436 │
│ rel   │ rel_ny… │  91.525 │
│ rel   │ rel_sc… │  11.492 │
│ event │ event_… │  65.405 │
│ event │ event_… │  38.835 │
│ event │ event_… │  67.409 │
│ event │ event_… │  41.953 │
│ absa  │ absa_1… │  70.852 │
│ absa  │ absa_1… │  61.635 │
│ absa  │ absa_1… │  67.660 │
│ absa  │ absa_1… │  71.414 │
└───────┴─────────┴─────────┘

mirror_outputs/Mirror_UIE_wPT_woOverlapV2
┏━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task  ┃ Dataset          ┃ Metric (%) ┃
┡━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent   │ ent_ace04_test   │     85.493 │
│ ent   │ ent_ace05_test   │     87.883 │
│ ent   │ ent_conll03_test │     91.378 │
│ rel   │ rel_ace05_test   │     57.431 │
│ rel   │ rel_conll04_test │     67.831 │
│ rel   │ rel_nyt_test     │     91.636 │
│ rel   │ rel_scierc_test  │     19.835 │
│ event │ event_ace05_tes… │     72.596 │
│ event │ event_ace05_tes… │     48.807 │
│ event │ event_casie_tes… │     66.073 │
│ event │ event_casie_tes… │     44.836 │
│ absa  │ absa_14res_test  │     74.842 │
│ absa  │ absa_14lap_test  │     65.126 │
│ absa  │ absa_15res_test  │     64.768 │
│ absa  │ absa_16res_test  │     73.867 │
└───────┴──────────────────┴────────────┘

mirror_outputs/Mirror_ExcludedPretrain_MultiTask
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task   ┃ Dataset                ┃ Metric (%) ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent    │ ent_conll03_test       │     91.840 │
│ rel    │ rel_conll04_test       │     72.391 │
│ event  │ event_ace05_test_tgg   │     68.241 │
│ event  │ event_ace05_test_arg   │     47.725 │
│ absa   │ absa_16res_test        │     75.310 │
└────────┴────────────────────────┴────────────┘

  mirror_outputs/MirrorLarge_SamplingPretrain_woOverlap
┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task    ┃ Dataset                         ┃ Metric (%) ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cls     │ cls_ag_news_train               │      0.000 │
│ cls     │ cls_anli_r1_train               │      0.000 │
│ cls     │ cls_anli_r2_train               │      0.000 │
│ cls     │ cls_anli_r3_train               │      0.000 │
│ cls     │ cls_arc_train                   │      0.000 │
│ cls     │ cls_cola_train                  │      0.000 │
│ cls     │ cls_cosmosqa_train              │      0.000 │
│ cls     │ cls_dbpedia_train               │      0.000 │
│ cls     │ cls_dream_train                 │      0.000 │
│ cls     │ cls_medqa_train                 │      0.000 │
│ cls     │ cls_mrpc_train                  │      0.000 │
│ cls     │ cls_openbookqa_train            │      0.000 │
│ cls     │ cls_race_train                  │      0.000 │
│ cls     │ cls_race-_trainc                │      0.000 │
│ cls     │ cls_sciq_train                  │      0.000 │
│ cls     │ cls_snli_train                  │      0.000 │
│ ent     │ ent_ace04_train                 │     86.606 │
│ ent     │ ent_ace05-uie_train             │     86.901 │
│ ent     │ ent_anatem_train                │     93.182 │
│ ent     │ ent_bc2gm_train                 │     86.888 │
│ ent     │ ent_bc4chemd_train              │     92.380 │
│ ent     │ ent_bc5cdr_train                │     93.862 │
│ ent     │ ent_broad_tweet_corpus_train    │     86.098 │
│ ent     │ ent_conll03_train               │     97.071 │
│ ent     │ ent_crossner_ai_train           │     70.653 │
│ ent     │ ent_crossner_literature_train   │     68.750 │
│ ent     │ ent_crossner_music_train        │     83.007 │
│ ent     │ ent_crossner_politics_train     │     84.950 │
│ ent     │ ent_crossner_science_train      │     71.975 │
│ ent     │ ent_fabner_train                │     81.018 │
│ ent     │ ent_findvehicle_train           │     97.073 │
│ ent     │ ent_genia_ner_train             │     82.966 │
│ ent     │ ent_harveyner_train             │     68.535 │
│ ent     │ ent_mit_movie_review_train      │     88.486 │
│ ent     │ ent_mit_restaurant_review_train │     84.813 │
│ ent     │ ent_multinerd_train             │     93.767 │
│ ent     │ ent_ncbidiease_train            │     92.506 │
│ ent     │ ent_ontonotes5_train            │     90.458 │
│ ent     │ ent_tweetner7_train             │     66.421 │ x
│ ent     │ ent_wikiann_en_train            │     87.184 │
│ ent     │ ent_wnut-16_train               │     74.102 │
│ event   │ event_ace05-evt-uie_train_tgg   │     71.371 │
│ event   │ event_ace05-evt-uie_train_arg   │     37.193 │
│ event   │ event_casie_train_tgg           │     49.564 │
│ event   │ event_casie_train_arg           │     15.333 │
│ event   │ event_phee_train_tgg            │     73.528 │
│ event   │ event_phee_train_arg            │     56.512 │
│ rel     │ rel_14lap_train                 │     62.124 │
│ rel     │ rel_14res_train                 │     66.458 │
│ rel     │ rel_15res_train                 │     77.044 │
│ rel     │ rel_16res_train                 │     73.037 │
│ rel     │ rel_ace05-rel-uie_train         │     27.854 │
│ rel     │ rel_conll04_train               │     58.760 │
│ rel     │ rel_nyt_multi_train             │     91.525 │
│ rel     │ rel_scierc_train                │     10.131 │
│ span_em │ span_subjqa_books_train         │     45.947 │
│ span_f1 │ span_subjqa_books_train         │     35.552 │
│ span_em │ span_subjqa_electronics_train   │     47.546 │
│ span_f1 │ span_subjqa_electronics_train   │     40.358 │
│ span_em │ span_subjqa_grocery_train       │     49.658 │
│ span_f1 │ span_subjqa_grocery_train       │     44.521 │
│ span_em │ span_subjqa_movies_train        │     42.560 │
│ span_f1 │ span_subjqa_movies_train        │     36.206 │
│ span_em │ span_subjqa_restaurants_train   │     50.195 │
│ span_f1 │ span_subjqa_restaurants_train   │     44.211 │
│ span_em │ span_subjqa_tripadvisor_train   │     38.765 │
│ span_f1 │ span_subjqa_tripadvisor_train   │     36.225 │
└─────────┴─────────────────────────────────┴────────────┘

mirror_outputs/Mirror_Pretrain_DataV1.
                 5_2
┏━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task ┃ Dataset        ┃ Metric (%) ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent  │ ent_movie      │     34.907 │
│ ent  │ ent_restaurant │     20.259 │
│ ent  │ ent_ai         │     23.115 │
│ ent  │ ent_literature │     36.566 │
│ ent  │ ent_music      │     33.716 │
│ ent  │ ent_politics   │     48.165 │
│ ent  │ ent_science    │     44.995 │
└──────┴────────────────┴────────────┘

mirror_outputs/Mirror_Pretrain_AllExcl
                uded_2
┏━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task ┃ Dataset        ┃ Metric (%) ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent  │ ent_movie      │     38.360 │
│ ent  │ ent_restaurant │     13.511 │
│ ent  │ ent_ai         │     42.998 │
│ ent  │ ent_literature │     41.638 │
│ ent  │ ent_music      │     56.655 │
│ ent  │ ent_politics   │     68.906 │
│ ent  │ ent_science    │     53.454 │
└──────┴────────────────┴────────────┘

best ckpt after the whole training
mirror_outputs/Mirror_Pretrain_AllExcl
                uded_2
┏━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task ┃ Dataset        ┃ Metric (%) ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent  │ ent_movie      │     39.201 │
│ ent  │ ent_restaurant │     16.318 │
│ ent  │ ent_ai         │     45.230 │
│ ent  │ ent_literature │     46.318 │
│ ent  │ ent_music      │     58.611 │
│ ent  │ ent_politics   │     67.303 │
│ ent  │ ent_science    │     54.837 │
└──────┴────────────────┴────────────┘

mirror_outputs/Mirror_Pretrain_DataV1.
           5_woInstruction
┏━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task ┃ Dataset        ┃ Metric (%) ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent  │ ent_movie      │     30.864 │
│ ent  │ ent_restaurant │     12.439 │
│ ent  │ ent_ai         │     32.317 │
│ ent  │ ent_literature │     40.048 │
│ ent  │ ent_music      │     40.600 │
│ ent  │ ent_politics   │     46.247 │
│ ent  │ ent_science    │     42.422 │
└──────┴────────────────┴────────────┘

    mirror_outputs/Mirror_Pretrain_AllExcluded_2
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task ┃ Dataset                       ┃ Metric (%) ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent  │ ent_mit_movie_review_icl      │     16.328 │
│ ent  │ ent_mit_restaurant_review_icl │      2.338 │
│ ent  │ ent_crossner_ai_icl           │     24.422 │
│ ent  │ ent_crossner_literature_icl   │     27.290 │
│ ent  │ ent_crossner_music_icl        │     35.316 │
│ ent  │ ent_crossner_politics_icl     │     40.492 │
│ ent  │ ent_crossner_science_icl      │     32.551 │
└──────┴───────────────────────────────┴────────────┘

mirror_outputs/Mirror_Pretrain_AllExc
               luded_2
┏━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task ┃ Dataset       ┃ Metric (%) ┃
┡━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ cls  │ cls_glue_cola │     63.908 │
│ cls  │ cls_glue_qqp  │     84.815 │
│ cls  │ cls_glue_mnli │     85.899 │
│ cls  │ cls_glue_sst2 │     93.585 │
│ cls  │ cls_glue_qnli │     91.616 │
│ cls  │ cls_glue_rte  │     85.921 │
│ cls  │ cls_glue_mrpc │     89.216 │
└──────┴───────────────┴────────────┘

      mirror_outputs/Mirror_Pretrain_AllExcluded_2
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task ┃ Dataset                           ┃ Metric (%) ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent  │ ent_crossner_ai_retrieval         │     28.845 │
│ ent  │ ent_crossner_literature_retrieval │     32.727 │
│ ent  │ ent_crossner_music_retrieval      │     38.181 │
│ ent  │ ent_crossner_politics_retrieval   │     47.693 │
│ ent  │ ent_crossner_science_retrieval    │     47.113 │
└──────┴───────────────────────────────────┴────────────┘

      mirror_outputs/Mirror_Pretrain_AllExcluded_2
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Task              ┃ Dataset              ┃ Metric (%) ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│ ent               │ ent_ace04_test       │     21.489 │
│ ent               │ ent_ace05_test       │     18.701 │
│ ent               │ ent_conll03_test     │     66.906 │
│ rel               │ rel_ace05_test       │      0.510 │
│ rel               │ rel_conll04_test     │      1.399 │
│ rel               │ rel_nyt_test         │     69.666 │
│ rel               │ rel_scierc_test      │      0.000 │
│ event             │ event_ace05_test_tgg │      3.991 │
│ event             │ event_ace05_test_arg │      0.000 │
│ event             │ event_casie_test_tgg │      2.128 │
│ event             │ event_casie_test_arg │      0.000 │
│ absa              │ absa_14res_test      │      0.000 │
│ absa              │ absa_14lap_test      │      0.000 │
│ absa              │ absa_15res_test      │      0.000 │
│ absa              │ absa_16res_test      │      0.000 │
│ discontinuous_ent │ discontinuous_ent    │     52.339 │
│ hyper_rel         │ hyper_rel            │      0.000 │
│ ent               │ ent_movie            │     39.237 │
│ ent               │ ent_restaurant       │     16.168 │
│ ent               │ ent_ai               │     45.912 │
│ ent               │ ent_literature       │     46.766 │
│ ent               │ ent_music            │     59.121 │
│ ent               │ ent_politics         │     67.274 │
│ ent               │ ent_science          │     54.418 │
│ span_em           │ span_squad2          │     40.351 │
│ span_f1           │ span_squad2          │     67.385 │
│ cls               │ cls_glue_cola        │     63.908 │
│ cls               │ cls_glue_qqp         │     84.845 │
│ cls               │ cls_glue_mnli        │     85.899 │
│ cls               │ cls_glue_sst2        │     93.585 │
│ cls               │ cls_glue_qnli        │     91.616 │
│ cls               │ cls_glue_rte         │     85.921 │
│ cls               │ cls_glue_mrpc        │     89.216 │
└───────────────────┴──────────────────────┴────────────┘
"""

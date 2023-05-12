from rex.utils.initialization import set_seed_and_log_path

from src.task import SchemaGuidedInstructBertTask

set_seed_and_log_path(log_path="eval.log")

task_dir = "mirror_outputs/InstructBert_TagSpan_DebertaV3Base_MergedUIEData"
task = SchemaGuidedInstructBertTask.from_taskdir(
    task_dir, load_best_model=True, initialize=False
)

data_pairs = [
    # fmt: off

    # [
    #     "rel_conll2004",
    #     "resources/Mirror/Tasks/RE/CoNLL2004/formatted/CoNLL2004_RE_test.jsonl",
    # ],
    # ["rel_gids", "resources/Mirror/Tasks/RE/GIDS/formatted/GIDS_test.jsonl"],
    # [
    #     "rel_nyt11hrl",
    #     "resources/Mirror/Tasks/RE/NYT11HRL/formatted/NYT11HRL_test.jsonl",
    # ],
    # ["rel_webnlg", "resources/Mirror/Tasks/RE/WebNLG/formatted/WebNLG_test.jsonl"],
    # ["rel_ace05", "resources/Mirror/Tasks/EE/ACE05-EN/ACE2005_oneie_RE_test.jsonl"],

    # UIE eval data
    ["ent_ace04_test", "resources/Mirror/v1.3/ent/en/ACE_2004/instructed/test.jsonl"],
    ["ent_conll03_test", "resources/Mirror/v1.3/ent/en/CoNLL2003/instructed/test.jsonl"],
    ["ent_ace05_test", "resources/Mirror/v1.3/ent/en/ACE05-EN-plus/instructed/test.jsonl"],
    ["rel_conll04_test", "resources/Mirror/v1.3/rel/en/CoNLL2004/instructed/CoNLL2004_RE_labelmap_test.jsonl"],
    ["rel_ace05_test", "resources/Mirror/v1.3/rel/en/ACE05-EN-plus/instructed/ACE2005_plus_RE_labelmap_test.jsonl"],
    ["rel_nyt_test", "resources/Mirror/v1.3/rel/en/NYT_multi/instructed/NYT_multi_test.jsonl"],
    ["rel_scierc_test", "resources/Mirror/v1.3/rel/en/sciERC/instructed/sciERC_test.jsonl"],
    ["event_ace05_test", "resources/Mirror/v1.3/event/en/ACE05-EN-plus/fixed_instructed/test.jsonl"],
    ["event_casie_test", "resources/Mirror/v1.3/event/en/CASIE/instructed/test.jsonl"],
    ["absa_14lap_test", "resources/Mirror/v1.3/rel/en/14lap/instructed/test.jsonl"],
    ["absa_14res_test", "resources/Mirror/v1.3/rel/en/14res/instructed/test.jsonl"],
    ["absa_15res_test", "resources/Mirror/v1.3/rel/en/15res/instructed/test.jsonl"],
    ["absa_16res_test", "resources/Mirror/v1.3/rel/en/16res/instructed/test.jsonl"],
    # fmt: on
]

for dname, fpath in data_pairs:
    task.data_manager.update_datapath(dname, fpath)
    task.eval(dname, verbose=True, dump=True, dump_middle=True)

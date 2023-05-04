from rex.utils.initialization import set_seed_and_log_path

from src.task import SchemaGuidedInstructBertTask

set_seed_and_log_path(log_path="eval.log")

task_dir = "outputs/InstructBert_TagSpan_DebertaV3Base_Rel_Merged202305022358v2"
task = SchemaGuidedInstructBertTask.from_taskdir(
    task_dir, load_best_model=True, initialize=False
)

data_pairs = [
    [
        "rel_conll2004",
        "resources/Mirror/Tasks/RE/CoNLL2004/formatted/CoNLL2004_RE_test.jsonl",
    ],
    ["rel_gids", "resources/Mirror/Tasks/RE/GIDS/formatted/GIDS_test.jsonl"],
    [
        "rel_nyt11hrl",
        "resources/Mirror/Tasks/RE/NYT11HRL/formatted/NYT11HRL_test.jsonl",
    ],
    ["rel_webnlg", "resources/Mirror/Tasks/RE/WebNLG/formatted/WebNLG_test.jsonl"],
    ["rel_ace05", "resources/Mirror/Tasks/EE/ACE05-EN/ACE2005_oneie_RE_test.jsonl"],
]

for dname, fpath in data_pairs:
    task.data_manager.update_datapath(dname, fpath)
    task.eval(dname, verbose=True, dump=True, dump_middle=True)

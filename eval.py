from rex.utils.initialization import set_seed_and_log_path

from src.task import MrcQaTask

set_seed_and_log_path(log_path="eval.log")

task_dir = "outputs/RobertaBase_data20230314v2"
task = MrcQaTask.from_taskdir(task_dir, load_best_model=True, initialize=False)

data_pairs = [
    ["cmrc_dev", "resources/MRC/cmrc2018/formatted/validation.jsonl"],
    ["drcd_dev", "resources/MRC/DRCD2018/formatted/dev.jsonl"],
]

for dname, fpath in data_pairs:
    task.data_manager.update_datapath(dname, fpath)
    task.eval(dname, verbose=True, dump=True, dump_middle=True)

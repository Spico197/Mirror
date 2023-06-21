import re
from collections import defaultdict

from rex.utils.io import load_jsonlines

files = """
resources/Mirror/v1.4_sampled_v3/cls/en/ag_news/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/ANLI/R1_processed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/ANLI/R2_processed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/ANLI/R3_processed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/ARC/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/CoLA/formated/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/CosmosQA/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/cos_e/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/dbpedia/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/DREAM/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/hellaswag/processed/hellaswag_train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/IMDB/formated/IMDB_train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/MedQA/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/MNLI/formated/MNLI_train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/MRPC/formated/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/MultiRC/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/OpenBookQA/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/QASC/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/QNLI/processed/QNLI_train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/QQP/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/RACE/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/RACE-C/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/ReClor/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/RTE/formated/RTE_train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/SciQ/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/SNLI/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/SST-2/instructed/SST-2_train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/Winogrande/new/train.jsonl \
resources/Mirror/v1.4_sampled_v3/cls/en/WNLI/processed/WNLI_train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/AnatEM/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/bc2gm/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/bc4chemd/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/bc5cdr/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/Broad_Tweet_Corpus/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/FabNER/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/FindVehicle/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/GENIA_NER/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/HarveyNER/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/MultiNERD/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/NCBIdiease/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/ontoNotes5/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/TweetNER7/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/WikiANN_en/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/ent/en/WNUT-16/train.jsonl \
resources/Mirror/v1.4_sampled_v3/event/en/PHEE/instructed/train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/ADE_corpus/instructed/ADE_corpus_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/FewRel/instructed/FewRel_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/GIDS/instructed/GIDS_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/kbp37/instructed/kbp37_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/New-York-Times-RE/instructed/New_York_Times_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/NYT11HRL/instructed/NYT11HRL_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/semeval/instructed/semeval_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/WebNLG/instructed/WebNLG_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/Wiki-ZSL/instructed/Wiki_ZSL_0_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/Wiki-ZSL/instructed/Wiki_ZSL_1_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/Wiki-ZSL/instructed/Wiki_ZSL_2_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/Wiki-ZSL/instructed/Wiki_ZSL_3_train.jsonl \
resources/Mirror/v1.4_sampled_v3/rel/en/Wiki-ZSL/instructed/Wiki_ZSL_4_train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/BiPaR/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/ms_marco_v2.1/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/newsqa/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/squad_v2/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/SubjQA/books/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/SubjQA/electronics/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/SubjQA/grocery/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/SubjQA/movies/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/SubjQA/restaurants/train.jsonl \
resources/Mirror/v1.4_sampled_v3/span/en/SubjQA/tripadvisor/train.jsonl
""".strip().split()


task2instructions = defaultdict(lambda: defaultdict(set))
task2instance = defaultdict(lambda: defaultdict(lambda: 0))
for file in files:
    re_obj = re.search(r".*/(cls|ent|event|rel|span)/en/(.*?)/.*\.jsonl", file.strip())
    if re_obj:
        task = re_obj.group(1)
        data_name = re_obj.group(2)
        data = load_jsonlines(file)
        instructions = [ins["instruction"] for ins in data]
        task2instructions[task][data_name].update(instructions)
        task2instance[task][data_name] += len(data)
# print(task2instructions)
for task in task2instructions:
    task_instructions = set()
    total_instance = 0
    for data_name in task2instructions[task]:
        task_instructions.update(task2instructions[task][data_name])
        total_instance += task2instance[task][data_name]
        print(
            f"{data_name} & {len(task2instructions[task][data_name])} & {task2instance[task][data_name]}"
        )
    print(task, len(task_instructions), total_instance)

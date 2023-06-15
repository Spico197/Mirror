import random

from rex.utils.io import dump_jsonlines, load_jsonlines

data = load_jsonlines(
    "resources/Mirror/v1.3/rel/en/T-REx/instructed/t-rex.udi.fix.jsonl"
)
rels = set()
for ins in data:
    rels.update(ins["schema"]["rel"])
rels = list(rels)

new_data = []
for ins in data:
    sampled_rels = random.choices(rels, k=4)
    ins["schema"]["rel"] += sampled_rels
    random.shuffle(ins["schema"]["rel"])
    new_data.append(ins)


for _ in range(20):
    random.shuffle(new_data)

dump_jsonlines(
    new_data, "resources/Mirror/v1.3/rel/en/T-REx/instructed/t-rex.udi.fix.neg.jsonl"
)

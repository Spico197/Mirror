import random

from rex.utils.io import dump_jsonlines, load_jsonlines

onto = load_jsonlines("resources/Mirror/v1.5/pretrain/OntoNotes5/train.jsonl")
nerd = load_jsonlines("resources/Mirror/v1.5/pretrain/MultiNERD/train.jsonl")

onto_ent_types = set()
nerd_ent_types = set()
for ins in onto:
    onto_ent_types.update(ins["schema"]["ent"])
for ins in nerd:
    nerd_ent_types.update(ins["schema"]["ent"])

new_onto = []
for ins in onto:
    ans_ent_types = set(ent["type"] for ent in ins["ans"]["ent"])
    sampled_in_ent_types = random.choices(list(onto_ent_types - ans_ent_types), k=3)
    sampled_out_ent_types = random.choices(
        list(nerd_ent_types - ans_ent_types - set(sampled_in_ent_types)), k=2
    )
    sampled_ent_types = (
        list(ans_ent_types) + sampled_in_ent_types + sampled_out_ent_types
    )
    random.shuffle(sampled_ent_types)
    ins["schema"]["ent"] = sampled_ent_types
    new_onto.append(ins)
dump_jsonlines(new_onto, "resources/Mirror/v1.5/pretrain/OntoNotes5/train.neg.jsonl")

new_nerd = []
for ins in nerd:
    ans_ent_types = set(ent["type"] for ent in ins["ans"]["ent"])
    sampled_in_ent_types = random.choices(list(nerd_ent_types - ans_ent_types), k=3)
    sampled_out_ent_types = random.choices(
        list(onto_ent_types - ans_ent_types - set(sampled_in_ent_types)), k=2
    )
    sampled_ent_types = (
        list(ans_ent_types) + sampled_in_ent_types + sampled_out_ent_types
    )
    random.shuffle(sampled_ent_types)
    ins["schema"]["ent"] = sampled_ent_types
    new_nerd.append(ins)
dump_jsonlines(new_nerd, "resources/Mirror/v1.5/pretrain/MultiNERD/train.neg.jsonl")

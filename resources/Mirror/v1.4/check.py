import os

from rex.utils.io import dump_jsonlines, load_jsonlines
from rex.utils.progress_bar import pbar


def start_endp1_span_is_valid(span: list, text: str):
    return len(span) == 2 and 0 <= span[0] < span[1] <= len(text)


def check_udi_instance(instance: dict):
    assert isinstance(instance["id"], str)
    assert isinstance(instance["instruction"], str)
    assert isinstance(instance["schema"], dict)
    for key in instance["schema"]:
        assert key in ["cls", "ent", "rel", "event"]
        if key in ["cls", "ent", "rel"]:
            assert isinstance(instance["schema"][key], list)
            assert all(
                isinstance(x, str) and len(x) > 0 for x in instance["schema"][key]
            )
        elif key == "event":
            assert isinstance(instance["schema"]["event"], dict)
            for event_type in instance["schema"]["event"]:
                assert isinstance(instance["schema"]["event"][event_type], list)
                assert all(
                    isinstance(x, str) and len(x) > 0
                    for x in instance["schema"]["event"][event_type]
                )
        else:
            raise ValueError
    assert isinstance(instance["ans"], dict)
    for key in instance["ans"]:
        assert key in ["cls", "ent", "rel", "event", "span"]
        if key == "cls":
            assert isinstance(instance["ans"]["cls"], list)
            assert all(
                isinstance(x, str) and len(x) > 0 for x in instance["ans"]["cls"]
            )
            assert all(x in instance["schema"]["cls"] for x in instance["ans"]["cls"])
        elif key == "ent":
            assert isinstance(instance["ans"]["ent"], list)
            assert all(isinstance(x, dict) for x in instance["ans"]["ent"])
            for ent in instance["ans"]["ent"]:
                assert isinstance(ent["type"], str)
                assert ent["type"] in instance["schema"]["ent"]
                assert isinstance(ent["text"], str)
                assert len(ent["text"]) > 0
                assert instance["text"][ent["span"][0] : ent["span"][1]] == ent["text"]
                assert isinstance(ent["span"], list)
                assert len(ent["span"]) == 2
                assert start_endp1_span_is_valid(ent["span"], instance["text"])
                assert all(isinstance(x, int) for x in ent["span"])
        elif key == "rel":
            assert isinstance(instance["ans"]["rel"], list)
            assert all(isinstance(x, dict) for x in instance["ans"]["rel"])
            for rel in instance["ans"]["rel"]:
                assert isinstance(rel["relation"], str)
                assert rel["relation"] in instance["schema"]["rel"]
                assert isinstance(rel["head"], dict)
                assert len(rel["head"]["text"]) > 0
                assert len(rel["tail"]["text"]) > 0
                assert start_endp1_span_is_valid(rel["head"]["span"], instance["text"])
                assert start_endp1_span_is_valid(rel["tail"]["span"], instance["text"])
                assert (
                    instance["text"][rel["head"]["span"][0] : rel["head"]["span"][1]]
                    == rel["head"]["text"]
                )
                assert isinstance(rel["tail"], dict)
                assert (
                    instance["text"][rel["tail"]["span"][0] : rel["tail"]["span"][1]]
                    == rel["tail"]["text"]
                )
        elif key == "event":
            assert isinstance(instance["ans"]["event"], list)
            assert all(isinstance(x, dict) for x in instance["ans"]["event"])
            for event in instance["ans"]["event"]:
                assert event["event_type"] in instance["schema"]["event"]
                assert isinstance(event["trigger"], dict)
                assert len(event["trigger"]["text"]) > 0
                assert event["trigger"]["text"] in instance["text"]
                assert start_endp1_span_is_valid(
                    event["trigger"]["span"], instance["text"]
                )
                assert (
                    instance["text"][
                        event["trigger"]["span"][0] : event["trigger"]["span"][1]
                    ]
                    == event["trigger"]["text"]
                )
                for arg in event["args"]:
                    assert (
                        arg["role"] in instance["schema"]["event"][event["event_type"]]
                    )
                    assert isinstance(arg["text"], str)
                    assert len(arg["text"]) > 0
                    assert start_endp1_span_is_valid(arg["span"], instance["text"])
                    assert (
                        instance["text"][arg["span"][0] : arg["span"][1]] == arg["text"]
                    )
        elif key == "span":
            assert isinstance(instance["ans"]["span"], list)
            assert all(isinstance(x, dict) for x in instance["ans"]["span"])
            for span in instance["ans"]["span"]:
                assert isinstance(span["text"], str)
                assert len(span["text"]) > 0
                assert start_endp1_span_is_valid(span["span"], instance["text"])
                assert (
                    instance["text"][span["span"][0] : span["span"][1]] == span["text"]
                )
        else:
            raise ValueError
    assert isinstance(instance["text"], str)
    assert isinstance(instance["bg"], str)
    for key in ["ent", "rel", "event"]:
        if instance["schema"].get(key):
            assert len(instance["text"]) > 0
    if "span" in instance["ans"]:
        assert len(instance["text"]) > 0
    assert instance["instruction"] or instance["text"] or instance["bg"]


def is_valid_udi_instance(instance: dict):
    ok = True
    try:
        check_udi_instance(instance)
    except:
        ok = False
    return ok


def find_jsonl(dir: str = "."):
    filepaths = []
    # Iterate over all directories and subdirectories in the current directory
    for root, subdirs, files in os.walk(dir):
        if not subdirs:
            # Filter the list of files to only include files that end with `jsonl`
            jsonl_files = {
                os.path.join(root, file) for file in files if file.endswith(".jsonl")
            }

            # Print the list of jsonl files
            for jsonl_file in jsonl_files:
                filepaths.append(jsonl_file)
    return filepaths


def main():
    filepaths = [
        "./cls/en/ag_news/instructed/test.jsonl",
        "./cls/en/ag_news/instructed/train.jsonl",
        "./cls/en/ANLI/R1_processed/train.jsonl",
        "./cls/en/ANLI/R1_processed/dev.jsonl",
        "./cls/en/ANLI/R1_processed/test.jsonl",
        "./cls/en/ANLI/R2_processed/dev.jsonl",
        "./cls/en/ANLI/R2_processed/train.jsonl",
        "./cls/en/ANLI/R2_processed/test.jsonl",
        "./cls/en/ANLI/R3_processed/dev.jsonl",
        "./cls/en/ANLI/R3_processed/train.jsonl",
        "./cls/en/ANLI/R3_processed/test.jsonl",
        "./cls/en/ARC/new/test.jsonl",
        "./cls/en/ARC/new/train.jsonl",
        "./cls/en/ARC/new/dev.jsonl",
        "./cls/en/CoLA/formated/test.jsonl",
        "./cls/en/CoLA/formated/train.jsonl",
        "./cls/en/CosmosQA/new/dev.jsonl",
        "./cls/en/CosmosQA/new/train.jsonl",
        "./cls/en/CosmosQA/new/test.jsonl",
        "./cls/en/cos_e/new/train.jsonl",
        "./cls/en/cos_e/new/dev.jsonl",
        "./cls/en/dbpedia/new/test.jsonl",
        "./cls/en/dbpedia/new/train.jsonl",
        "./cls/en/DREAM/new/test.jsonl",
        "./cls/en/DREAM/new/dev.jsonl",
        "./cls/en/DREAM/new/train.jsonl",
        "./cls/en/hellaswag/processed/hellaswag_train.jsonl",
        "./cls/en/hellaswag/processed/hellaswag_dev.jsonl",
        "./cls/en/IMDB/formated/IMDB_train.jsonl",
        "./cls/en/IMDB/formated/IMDB_dev.jsonl",
        "./cls/en/MCTest/new/dev_500.jsonl",
        "./cls/en/MCTest/new/train_500.jsonl",
        "./cls/en/MCTest/new/dev_160.jsonl",
        "./cls/en/MCTest/new/test_500.jsonl",
        "./cls/en/MCTest/new/train_160.jsonl",
        "./cls/en/MCTest/new/test_160.jsonl",
        "./cls/en/MedQA/new/test.jsonl",
        "./cls/en/MedQA/new/train.jsonl",
        "./cls/en/MedQA/new/dev.jsonl",
        "./cls/en/MNLI/formated/MNLI_train.jsonl",
        "./cls/en/MNLI/formated/MNLI_dev.jsonl",
        "./cls/en/MRPC/formated/train.jsonl",
        "./cls/en/MRPC/formated/dev.jsonl",
        "./cls/en/MRPC/formated/test.jsonl",
        "./cls/en/MultiRC/instructed/train.jsonl",
        "./cls/en/MultiRC/instructed/dev.jsonl",
        "./cls/en/OpenBookQA/new/test.jsonl",
        "./cls/en/OpenBookQA/new/train.jsonl",
        "./cls/en/OpenBookQA/new/dev.jsonl",
        "./cls/en/QASC/new/train.jsonl",
        "./cls/en/QASC/new/dev.jsonl",
        "./cls/en/QNLI/processed/QNLI_dev.jsonl",
        "./cls/en/QNLI/processed/QNLI_train.jsonl",
        "./cls/en/QQP/new/train.jsonl",
        "./cls/en/QQP/new/dev.jsonl",
        "./cls/en/RACE/instructed/dev.jsonl",
        "./cls/en/RACE/instructed/test.jsonl",
        "./cls/en/RACE/instructed/train.jsonl",
        "./cls/en/RACE-C/new/dev.jsonl",
        "./cls/en/RACE-C/new/test.jsonl",
        "./cls/en/RACE-C/new/train.jsonl",
        "./cls/en/ReClor/new/dev.jsonl",
        "./cls/en/ReClor/new/train.jsonl",
        "./cls/en/RTE/formated/RTE_dev.jsonl",
        "./cls/en/RTE/formated/RTE_train.jsonl",
        "./cls/en/SciQ/instructed/test.jsonl",
        "./cls/en/SciQ/instructed/dev.jsonl",
        "./cls/en/SciQ/instructed/train.jsonl",
        "./cls/en/SNLI/instructed/dev.jsonl",
        "./cls/en/SNLI/instructed/train.jsonl",
        "./cls/en/SNLI/instructed/test.jsonl",
        "./cls/en/SST-2/instructed/SST-2_dev.jsonl",
        "./cls/en/SST-2/instructed/SST-2_train.jsonl",
        "./cls/en/Winogrande/new/train.jsonl",
        "./cls/en/Winogrande/new/dev.jsonl",
        "./cls/en/WNLI/processed/WNLI_train.jsonl",
        "./cls/en/WNLI/processed/WNLI_dev.jsonl",
        "./ent/en/ACE05-EN/instructed/dev.jsonl",
        "./ent/en/ACE05-EN/instructed/test.jsonl",
        "./ent/en/ACE05-EN/instructed/train.jsonl",
        "./ent/en/ACE05-EN-plus/instructed/test.jsonl",
        "./ent/en/ACE05-EN-plus/instructed/train.jsonl",
        "./ent/en/ACE05-EN-plus/instructed/dev.jsonl",
        "./ent/en/ACE_2004/instructed/test.jsonl",
        "./ent/en/ACE_2004/instructed/dev.jsonl",
        "./ent/en/ACE_2004/instructed/train.jsonl",
        "./ent/en/AnatEM/instructed/dev.jsonl",
        "./ent/en/AnatEM/instructed/test.jsonl",
        "./ent/en/AnatEM/instructed/train.jsonl",
        "./ent/en/bc2gm/instructed/train.jsonl",
        "./ent/en/bc2gm/instructed/dev.jsonl",
        "./ent/en/bc2gm/instructed/test.jsonl",
        "./ent/en/bc4chemd/instructed/train.jsonl",
        "./ent/en/bc4chemd/instructed/dev.jsonl",
        "./ent/en/bc4chemd/instructed/test.jsonl",
        "./ent/en/bc5cdr/instructed/test.jsonl",
        "./ent/en/bc5cdr/instructed/dev.jsonl",
        "./ent/en/bc5cdr/instructed/train.jsonl",
        "./ent/en/Broad_Tweet_Corpus/instructed/train.jsonl",
        "./ent/en/Broad_Tweet_Corpus/instructed/test.jsonl",
        "./ent/en/Broad_Tweet_Corpus/instructed/dev.jsonl",
        "./ent/en/CoNLL2003/instructed/test.jsonl",
        "./ent/en/CoNLL2003/instructed/dev.jsonl",
        "./ent/en/CoNLL2003/instructed/train.jsonl",
        "./ent/en/CrossNER_AI/instructed/dev.jsonl",
        "./ent/en/CrossNER_AI/instructed/train.jsonl",
        "./ent/en/CrossNER_AI/instructed/test.jsonl",
        "./ent/en/CrossNER_literature/instructed/train.jsonl",
        "./ent/en/CrossNER_literature/instructed/dev.jsonl",
        "./ent/en/CrossNER_literature/instructed/test.jsonl",
        "./ent/en/CrossNER_music/instructed/test.jsonl",
        "./ent/en/CrossNER_music/instructed/dev.jsonl",
        "./ent/en/CrossNER_music/instructed/train.jsonl",
        "./ent/en/CrossNER_politics/instructed/dev.jsonl",
        "./ent/en/CrossNER_politics/instructed/train.jsonl",
        "./ent/en/CrossNER_politics/instructed/test.jsonl",
        "./ent/en/CrossNER_science/instructed/dev.jsonl",
        "./ent/en/CrossNER_science/instructed/train.jsonl",
        "./ent/en/CrossNER_science/instructed/test.jsonl",
        "./ent/en/FabNER/instructed/dev.jsonl",
        "./ent/en/FabNER/instructed/test.jsonl",
        "./ent/en/FabNER/instructed/train.jsonl",
        "./ent/en/FindVehicle/instructed/train.jsonl",
        "./ent/en/FindVehicle/instructed/test.jsonl",
        "./ent/en/FindVehicle/instructed/dev.jsonl",
        "./ent/en/GENIA_NER/instructed/train.jsonl",
        "./ent/en/GENIA_NER/instructed/dev.jsonl",
        "./ent/en/GENIA_NER/instructed/test.jsonl",
        "./ent/en/HarveyNER/instructed/dev.jsonl",
        "./ent/en/HarveyNER/instructed/test.jsonl",
        "./ent/en/HarveyNER/instructed/train.jsonl",
        "./ent/en/MIT_MOVIE_Review/instructed/test.jsonl",
        "./ent/en/MIT_MOVIE_Review/instructed/dev.jsonl",
        "./ent/en/MIT_MOVIE_Review/instructed/train.jsonl",
        "./ent/en/MIT_Restaurant_Review/instructed/train.jsonl",
        "./ent/en/MIT_Restaurant_Review/instructed/dev.jsonl",
        "./ent/en/MIT_Restaurant_Review/instructed/test.jsonl",
        "./ent/en/MultiNERD/instructed/test.jsonl",
        "./ent/en/MultiNERD/instructed/train.jsonl",
        "./ent/en/MultiNERD/instructed/dev.jsonl",
        "./ent/en/NCBIdiease/instructed/test.jsonl",
        "./ent/en/NCBIdiease/instructed/dev.jsonl",
        "./ent/en/NCBIdiease/instructed/train.jsonl",
        "./ent/en/ontoNotes5/instructed/train.jsonl",
        "./ent/en/ontoNotes5/instructed/dev.jsonl",
        "./ent/en/ontoNotes5/instructed/test.jsonl",
        "./ent/en/TweetNER7/instructed/test.jsonl",
        "./ent/en/TweetNER7/instructed/dev.jsonl",
        "./ent/en/TweetNER7/instructed/train.jsonl",
        "./ent/en/WikiANN_en/instructed/test.jsonl",
        "./ent/en/WikiANN_en/instructed/dev.jsonl",
        "./ent/en/WikiANN_en/instructed/train.jsonl",
        "./ent/en/WNUT-16/train.jsonl",
        "./ent/en/WNUT-16/dev.jsonl",
        "./ent/en/WNUT-16/test.jsonl",
        "./event/en/ACE05-EN/instructed/test.jsonl",
        "./event/en/ACE05-EN/instructed/dev.jsonl",
        "./event/en/ACE05-EN/instructed/train.jsonl",
        "./event/en/ACE05-EN-plus/fixed_instructed/train.jsonl",
        "./event/en/ACE05-EN-plus/fixed_instructed/dev.jsonl",
        "./event/en/ACE05-EN-plus/fixed_instructed/test.jsonl",
        "./event/en/CASIE/instructed/train.jsonl",
        "./event/en/CASIE/instructed/dev.jsonl",
        "./event/en/CASIE/instructed/test.jsonl",
        "./event/en/PHEE/instructed/test.jsonl",
        "./event/en/PHEE/instructed/train.jsonl",
        "./event/en/PHEE/instructed/dev.jsonl",
        "./rel/en/14lap/instructed/dev.jsonl",
        "./rel/en/14lap/instructed/test.jsonl",
        "./rel/en/14lap/instructed/train.jsonl",
        "./rel/en/14res/instructed/dev.jsonl",
        "./rel/en/14res/instructed/train.jsonl",
        "./rel/en/14res/instructed/test.jsonl",
        "./rel/en/15res/instructed/train.jsonl",
        "./rel/en/15res/instructed/test.jsonl",
        "./rel/en/15res/instructed/dev.jsonl",
        "./rel/en/16res/instructed/train.jsonl",
        "./rel/en/16res/instructed/dev.jsonl",
        "./rel/en/16res/instructed/test.jsonl",
        "./rel/en/ACE05-EN/instructed/ACE2005_oneie_RE_labelmap_test.jsonl",
        "./rel/en/ACE05-EN/instructed/ACE2005_oneie_RE_labelmap_train.jsonl",
        "./rel/en/ACE05-EN/instructed/ACE2005_oneie_RE_labelmap_dev.jsonl",
        "./rel/en/ACE05-EN-plus/instructed/ACE2005_plus_RE_labelmap_train.jsonl",
        "./rel/en/ACE05-EN-plus/instructed/ACE2005_plus_RE_labelmap_test.jsonl",
        "./rel/en/ACE05-EN-plus/instructed/ACE2005_plus_RE_labelmap_dev.jsonl",
        "./rel/en/ADE_corpus/instructed/ADE_corpus_train.jsonl",
        "./rel/en/ADE_corpus/instructed/ADE_corpus_dev.jsonl",
        "./rel/en/ADE_corpus/instructed/ADE_corpus_test.jsonl",
        "./rel/en/CoNLL2004/instructed/CoNLL2004_RE_labelmap_train.jsonl",
        "./rel/en/CoNLL2004/instructed/CoNLL2004_RE_labelmap_test.jsonl",
        "./rel/en/CoNLL2004/instructed/CoNLL2004_RE_labelmap_dev.jsonl",
        "./rel/en/FewRel/instructed/FewRel_dev.jsonl",
        "./rel/en/FewRel/instructed/FewRel_train.jsonl",
        "./rel/en/GIDS/instructed/GIDS_test.jsonl",
        "./rel/en/GIDS/instructed/GIDS_dev.jsonl",
        "./rel/en/GIDS/instructed/GIDS_train.jsonl",
        "./rel/en/kbp37/instructed/kbp37_train.jsonl",
        "./rel/en/kbp37/instructed/kbp37_dev.jsonl",
        "./rel/en/kbp37/instructed/kbp37_test.jsonl",
        "./rel/en/New-York-Times-RE/instructed/New_York_Times_test.jsonl",
        "./rel/en/New-York-Times-RE/instructed/New_York_Times_train.jsonl",
        "./rel/en/NYT11HRL/instructed/NYT11HRL_test.jsonl",
        "./rel/en/NYT11HRL/instructed/NYT11HRL_test-plus.jsonl",
        "./rel/en/NYT11HRL/instructed/NYT11HRL_train.jsonl",
        "./rel/en/NYT_multi/instructed/NYT_multi_dev.jsonl",
        "./rel/en/NYT_multi/instructed/NYT_multi_train.jsonl",
        "./rel/en/NYT_multi/instructed/NYT_multi_test.jsonl",
        "./rel/en/sciERC/instructed/sciERC_test.jsonl",
        "./rel/en/sciERC/instructed/sciERC_dev.jsonl",
        "./rel/en/sciERC/instructed/sciERC_train.jsonl",
        "./rel/en/semeval/instructed/semeval_test.jsonl",
        "./rel/en/semeval/instructed/semeval_train.jsonl",
        "./rel/en/T-REx/instructed/t-rex.udi.fix.jsonl",
        "./rel/en/WebNLG/instructed/WebNLG_dev.jsonl",
        "./rel/en/WebNLG/instructed/WebNLG_test.jsonl",
        "./rel/en/WebNLG/instructed/WebNLG_train.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_0_train.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_2_train.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_0_test.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_4_test.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_2_test.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_1_train.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_1_test.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_3_train.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_3_test.jsonl",
        "./rel/en/Wiki-ZSL/instructed/Wiki_ZSL_4_train.jsonl",
        "./span/en/BiPaR/dev.jsonl",
        "./span/en/BiPaR/test.jsonl",
        "./span/en/BiPaR/train.jsonl",
        "./span/en/ms_marco_v2.1/train.jsonl",
        "./span/en/ms_marco_v2.1/dev.jsonl",
        "./span/en/newsqa/dev.jsonl",
        "./span/en/newsqa/train.jsonl",
        "./span/en/squad_v2/train.jsonl",
        "./span/en/squad_v2/dev.jsonl",
        "./span/en/SubjQA/books/dev.jsonl",
        "./span/en/SubjQA/books/test.jsonl",
        "./span/en/SubjQA/books/train.jsonl",
        "./span/en/SubjQA/electronics/dev.jsonl",
        "./span/en/SubjQA/electronics/test.jsonl",
        "./span/en/SubjQA/grocery/train.jsonl",
        "./span/en/SubjQA/grocery/dev.jsonl",
        "./span/en/SubjQA/grocery/test.jsonl",
        "./span/en/SubjQA/movies/train.jsonl",
        "./span/en/SubjQA/movies/test.jsonl",
        "./span/en/SubjQA/movies/dev.jsonl",
        "./span/en/SubjQA/restaurants/dev.jsonl",
        "./span/en/SubjQA/restaurants/train.jsonl",
        "./span/en/SubjQA/restaurants/test.jsonl",
        "./span/en/SubjQA/tripadvisor/dev.jsonl",
        "./span/en/SubjQA/tripadvisor/train.jsonl",
        "./span/en/SubjQA/tripadvisor/test.jsonl",
    ]
    # filepaths = find_jsonl(".")
    bar = pbar(filepaths)
    for filepath in bar:
        data = load_jsonlines(filepath)
        data_ok = True
        for ins in data:
            if not ins["instruction"]:
                bar.write(f"No Instruction: {filepath}")
                break
            ok = True
            # ok = is_valid_udi_instance(ins)
            try:
                check_udi_instance(ins)
            except Exception as err:
                ok = False
                bar.write(f"❌ {filepath}")
                raise err
            if not ok:
                data_ok = False
                break
        if not data_ok:
            bar.write(f"❌ {filepath}")


def filter_snli():
    for dname in ["train", "dev", "test"]:
        filepath = f"cls/en/SNLI/processed/SNLI_{dname}.jsonl"
        data = load_jsonlines(filepath)
        filtered = [ins for ins in filter(lambda ins: is_valid_udi_instance(ins), data)]
        dump_jsonlines(filtered, f"cls/en/SNLI/instructed/{dname}.jsonl")


if __name__ == "__main__":
    main()
    # filter_snli()

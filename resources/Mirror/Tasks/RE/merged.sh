FolderName="resources/Mirror/Tasks/RE/merged-20230502-2358-v2-woADE"

mkdir -p $FolderName

cat \
    resources/Mirror/Tasks/RE/CoNLL2004/formatted/CoNLL2004_RE_train.jsonl \
    resources/Mirror/Tasks/RE/GIDS/formatted/GIDS_train.jsonl \
    resources/Mirror/Tasks/RE/NYT11HRL/formatted/NYT11HRL_train.jsonl \
    resources/Mirror/Tasks/RE/WebNLG/formatted/WebNLG_train.jsonl \
    resources/Mirror/Tasks/EE/ACE05-EN/ACE2005_oneie_RE_train.jsonl \
        > "$FolderName/train.jsonl"
    # resources/Mirror/Tasks/RE/ADE_corpus/formatted/ADE_corpus_train.jsonl \

cat \
    resources/Mirror/Tasks/RE/CoNLL2004/formatted/CoNLL2004_RE_dev.jsonl \
    resources/Mirror/Tasks/RE/GIDS/formatted/GIDS_dev.jsonl \
    resources/Mirror/Tasks/RE/WebNLG/formatted/WebNLG_dev.jsonl \
    resources/Mirror/Tasks/EE/ACE05-EN/ACE2005_oneie_RE_dev.jsonl \
        > "$FolderName/dev.jsonl"
    # resources/Mirror/Tasks/RE/ADE_corpus/formatted/ADE_corpus_dev.jsonl \

cat \
    resources/Mirror/Tasks/RE/CoNLL2004/formatted/CoNLL2004_RE_test.jsonl \
    resources/Mirror/Tasks/RE/GIDS/formatted/GIDS_test.jsonl \
    resources/Mirror/Tasks/RE/NYT11HRL/formatted/NYT11HRL_test-plus.jsonl \
    resources/Mirror/Tasks/RE/NYT11HRL/formatted/NYT11HRL_test.jsonl \
    resources/Mirror/Tasks/RE/WebNLG/formatted/WebNLG_test.jsonl \
    resources/Mirror/Tasks/EE/ACE05-EN/ACE2005_oneie_RE_test.jsonl \
        > "$FolderName/test.jsonl"
    # resources/Mirror/Tasks/RE/ADE_corpus/formatted/ADE_corpus_test.jsonl \

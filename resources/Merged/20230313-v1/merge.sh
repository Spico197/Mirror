cat \
    resources/MRC/C3/formatted/m-train.jsonl \
    resources/MRC/C3/formatted/d-train.jsonl \
    resources/MRC/CHID/formatted/train.jsonl \
    resources/MRC/cmrc2018/formatted/train.jsonl \
    resources/MRC/DuReader-Checklist/formatted/train.jsonl \
    resources/MRC/DuReader-Robust/formatted/train.jsonl \
    resources/MRC/DuReader-yesno/formatted/train.jsonl \
    resources/RE/COER/coer.jsonl \
    resources/Classification/IFLYTEK/formatted/train.jsonl \
    resources/Classification/TNEWS/formatted/train.jsonl \
    resources/Matching/AFQMC/formatted/train.jsonl \
    resources/Matching/OCNLI/formatted/train.jsonl \
    resources/Coreference/CLUEWSC2020/formatted/train.jsonl \
    resources/Keyword/CSL/formatted/train.cat.jsonl \
    resources/NER/msra/mrc/train.jsonl > resources/Merged/20230313-v1/train.jsonl

cat \
    resources/Classification/IFLYTEK/formatted/dev.jsonl \
    resources/Classification/TNEWS/formatted/dev.jsonl \
    resources/Coreference/CLUEWSC2020/formatted/dev.jsonl \
    resources/Keyword/CSL/formatted/dev.jsonl \
    resources/Matching/AFQMC/formatted/dev.jsonl \
    resources/Matching/OCNLI/formatted/dev.jsonl \
    resources/MRC/C3/formatted/d-dev.jsonl \
    resources/MRC/C3/formatted/m-dev.jsonl \
    resources/MRC/CHID/formatted/dev.jsonl \
    resources/MRC/cmrc2018/formatted/validation.jsonl \
    resources/MRC/DuReader-Checklist/formatted/dev.jsonl \
    resources/MRC/DuReader-Robust/formatted/dev.jsonl \
    resources/MRC/DuReader-yesno/formatted/dev.jsonl > resources/Merged/20230313-v1/dev.jsonl

cat \
    resources/Keyword/CSL/formatted/test.jsonl \
    resources/MRC/cmrc2018/formatted/test.jsonl \
    resources/NER/msra/mrc/test.jsonl > resources/Merged/20230313-v1/test.jsonl

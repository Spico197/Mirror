OUTPUT_DIR="resources/Merged/20230323-v2-MRC-ExtractiveOnly"

cat \
    resources/MRC/DRCD2018/formatted/train.jsonl \
    resources/MRC/cmrc2018/formatted/train.jsonl \
    resources/MRC/DuReader-Checklist/formatted/train.jsonl \
    resources/MRC/DuReader-Robust/formatted/train.jsonl > ${OUTPUT_DIR}/train.jsonl

cat \
    resources/MRC/DRCD2018/formatted/dev.jsonl \
    resources/MRC/cmrc2018/formatted/validation.jsonl \
    resources/MRC/DuReader-Checklist/formatted/dev.jsonl \
    resources/MRC/DuReader-Robust/formatted/dev.jsonl > ${OUTPUT_DIR}/dev.jsonl

cat \
    resources/MRC/cmrc2018/formatted/test.jsonl \
    resources/MRC/DRCD2018/formatted/test.jsonl > ${OUTPUT_DIR}/test.jsonl


for (( i = 1; i <= 50; i++ ))
do
    echo "shuf ${i}"
    shuf ${OUTPUT_DIR}/train.jsonl -o ${OUTPUT_DIR}/train.jsonl
done
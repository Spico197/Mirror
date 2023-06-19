OUTPUT_DIR="resources/Mirror/v1.5/merged/t-rex-200k-woInstruction"

mkdir -p $OUTPUT_DIR

head -n 200000 resources/Mirror/v1.5/pretrain/T-REx/t-rex.udi.fix.neg.jsonl > resources/Mirror/v1.5/pretrain/T-REx/t-rex.udi.fix.neg.200k.jsonl

cat \
    resources/Mirror/v1.5/pretrain/T-REx/t-rex.udi.fix.neg.200k.jsonl \
    resources/Mirror/v1.5/pretrain/OntoNotes5/train.neg.jsonl \
    resources/Mirror/v1.5/pretrain/MultiNERD/train.neg.jsonl > ${OUTPUT_DIR}/train.jsonl

cat \
    resources/Mirror/v1.5/pretrain/MultiNERD/dev.jsonl \
    resources/Mirror/v1.5/pretrain/OntoNotes5/dev.jsonl > ${OUTPUT_DIR}/dev.jsonl

cat \
    resources/Mirror/v1.5/pretrain/MultiNERD/test.jsonl \
    resources/Mirror/v1.5/pretrain/OntoNotes5/test.jsonl > ${OUTPUT_DIR}/test.jsonl


for (( i = 1; i <= 50; i++ ))
do
    echo "shuf ${i}"
    shuf ${OUTPUT_DIR}/train.jsonl -o ${OUTPUT_DIR}/train.jsonl
done

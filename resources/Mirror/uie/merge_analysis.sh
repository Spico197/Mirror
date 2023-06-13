MERGED_DIR="resources/Mirror/uie/merged_analysis"

mkdir $MERGED_DIR

cat \
    resources/Mirror/uie/ent/conll03/train.jsonl \
    resources/Mirror/uie/rel/conll04/train.jsonl \
    resources/Mirror/uie/event/ace05-evt/train.jsonl \
    resources/Mirror/uie/absa/16res/train.jsonl \
    > $MERGED_DIR/train.jsonl

cat \
    resources/Mirror/uie/ent/conll03/dev.jsonl \
    resources/Mirror/uie/rel/conll04/dev.jsonl \
    resources/Mirror/uie/event/ace05-evt/dev.jsonl \
    resources/Mirror/uie/absa/16res/dev.jsonl \
    > $MERGED_DIR/dev.jsonl

cat \
    resources/Mirror/uie/ent/conll03/test.jsonl \
    resources/Mirror/uie/rel/conll04/test.jsonl \
    resources/Mirror/uie/event/ace05-evt/test.jsonl \
    resources/Mirror/uie/absa/16res/test.jsonl \
    > $MERGED_DIR/test.jsonl


for (( i = 1; i <= 20; i++ ))
do
    echo "shuf ${i}"
    shuf ${MERGED_DIR}/train.jsonl -o ${MERGED_DIR}/train.jsonl
done

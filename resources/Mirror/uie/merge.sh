MERGED_DIR="resources/Mirror/uie/merged"

mkdir $MERGED_DIR

cat \
    resources/Mirror/uie/ent/ace04/train.jsonl \
    resources/Mirror/uie/ent/ace05/train.jsonl \
    resources/Mirror/uie/ent/conll03/train.jsonl \
    resources/Mirror/uie/absa/14lap/train.jsonl \
    resources/Mirror/uie/absa/14res/train.jsonl \
    resources/Mirror/uie/absa/15res/train.jsonl \
    resources/Mirror/uie/absa/16res/train.jsonl \
    resources/Mirror/uie/rel/ace05-rel/train.jsonl \
    resources/Mirror/uie/rel/conll04/train.jsonl \
    resources/Mirror/uie/rel/nyt/train.jsonl \
    resources/Mirror/uie/rel/scierc/train.jsonl \
    resources/Mirror/uie/event/ace05-evt/train.jsonl \
    resources/Mirror/uie/event/casie/train.jsonl \
    > $MERGED_DIR/train.jsonl

cat \
    resources/Mirror/uie/absa/14lap/dev.jsonl \
    resources/Mirror/uie/absa/14res/dev.jsonl \
    resources/Mirror/uie/absa/15res/dev.jsonl \
    resources/Mirror/uie/absa/16res/dev.jsonl \
    resources/Mirror/uie/ent/ace04/dev.jsonl \
    resources/Mirror/uie/ent/ace05/dev.jsonl \
    resources/Mirror/uie/ent/conll03/dev.jsonl \
    resources/Mirror/uie/event/ace05-evt/dev.jsonl \
    resources/Mirror/uie/event/casie/dev.jsonl \
    resources/Mirror/uie/rel/ace05-rel/dev.jsonl \
    resources/Mirror/uie/rel/conll04/dev.jsonl \
    resources/Mirror/uie/rel/nyt/dev.jsonl \
    resources/Mirror/uie/rel/scierc/dev.jsonl \
    > $MERGED_DIR/dev.jsonl

cat \
    resources/Mirror/uie/absa/14lap/test.jsonl \
    resources/Mirror/uie/absa/14res/test.jsonl \
    resources/Mirror/uie/absa/15res/test.jsonl \
    resources/Mirror/uie/absa/16res/test.jsonl \
    resources/Mirror/uie/ent/ace04/test.jsonl \
    resources/Mirror/uie/ent/ace05/test.jsonl \
    resources/Mirror/uie/ent/conll03/test.jsonl \
    resources/Mirror/uie/event/ace05-evt/test.jsonl \
    resources/Mirror/uie/event/casie/test.jsonl \
    resources/Mirror/uie/rel/ace05-rel/test.jsonl \
    resources/Mirror/uie/rel/conll04/test.jsonl \
    resources/Mirror/uie/rel/nyt/test.jsonl \
    resources/Mirror/uie/rel/scierc/test.jsonl \
    > $MERGED_DIR/test.jsonl


for (( i = 1; i <= 20; i++ ))
do
    echo "shuf ${i}"
    shuf ${MERGED_DIR}/train.jsonl -o ${MERGED_DIR}/train.jsonl
done

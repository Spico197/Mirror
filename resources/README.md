# Data Preprocessing

## Universal Format

### NER

One json at a line: [jsonl](https://jsonlines.org/).

```json
{
    "id": "an id",
    "tokens": ["佟", "湘", "玉", "是", "李", "大", "嘴", "的", "老", "板", "娘"],
    "ents": [
        {"type": "PER", "index": [0, 1, 2]},
        {"type": "PER", "index": [4, 5, 6]}
    ]
}
```

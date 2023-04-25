# udi-v1: universal data interface
{
    "id": "semeval.train.0",
    "instruction": "instruction text",
    "schema": {
        "cls": ["class1", "class2"],
        "ent": ["person", "location"],
        "rel": ["birth in", "study in"],
        "event": {
            "event type (attack)": ["roles like instrument", "attacker"],
            "another type": ["role", "role"],
        },
    },
    "ans": {
        "cls": ["class1"],
        "ent": [
            {"type": "person", "text": "Tong", "span": [0, 4]}
        ],  # span: [start, end + 1]
        "rel": [
            {
                "relation": "study in",
                "head": {"text": "Tong", "span": [0, 4]},
                "tail": {"text": "SUDA", "span": [5, 9]},
            }
        ],
        "event": [
            {
                "event_type": "attack",
                "trigger": {"text": "hit", "span": [6, 9]},
                "args": [{"role": "instrument", "text": "ax", "span": [8, 10]}],
            }
        ],
        "span": [{"text": "machine learning", "span": [16, 32]}],
    },
    # DONE: whether or not to concatenate instruction with text (v2)
    "text": "plain text",
    "bg": "background text",
}

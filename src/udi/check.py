from rex.utils.io import load_jsonlines


def check_udi_instance(instance: dict):
    assert isinstance(instance["id"], str)
    assert isinstance(instance["instruction"], str)
    assert isinstance(instance["schema"], dict)
    for key in instance["schema"]:
        assert key in ["cls", "ent", "rel", "event"]
        if key in ["cls", "ent", "rel"]:
            assert isinstance(instance["schema"][key], list) and all(
                isinstance(x, str) for x in instance["schema"][key]
            )
        elif key == "event":
            assert isinstance(instance["schema"][key], dict)
            for event_type in instance["schema"][key]:
                assert isinstance(instance["schema"][key][event_type], list) and all(
                    isinstance(x, str) for x in instance["schema"][key][event_type]
                )
        else:
            raise ValueError
    assert isinstance(instance["ans"], dict)
    for key in instance["ans"]:
        assert key in ["cls", "ent", "rel", "event", "span"]
        if key == "cls":
            assert isinstance(instance["ans"][key], list) and all(
                isinstance(x, str) for x in instance["ans"][key]
            )
        elif key == "ent":
            assert isinstance(instance["ans"][key], list) and all(
                isinstance(x, dict) for x in instance["ans"][key]
            )
            for ent in instance["ans"][key]:
                assert (
                    isinstance(ent["type"], str)
                    and ent["type"] in instance["schema"]["ent"]
                )
                assert (
                    isinstance(ent["text"], str)
                    and instance["text"][ent["span"][0] : ent["span"][1]] == ent["text"]
                )
                assert (
                    isinstance(ent["span"], list)
                    and len(ent["span"]) == 2
                    and all(isinstance(x, int) for x in ent["span"])
                )
        elif key == "rel":
            assert isinstance(instance["ans"][key], list) and all(
                isinstance(x, dict) for x in instance["ans"][key]
            )
            for rel in instance["ans"][key]:
                assert (
                    isinstance(rel["relation"], str)
                    and rel["relation"] in instance["schema"]["rel"]
                )
                assert (
                    isinstance(rel["head"], dict)
                    and instance["text"][
                        rel["head"]["span"][0] : rel["head"]["span"][1]
                    ]
                    == rel["head"]["text"]
                )
                assert (
                    isinstance(rel["tail"], dict)
                    and instance["text"][
                        rel["tail"]["span"][0] : rel["tail"]["span"][1]
                    ]
                    == rel["tail"]["text"]
                )
        elif key == "event":
            assert isinstance(instance["ans"][key], list) and all(
                isinstance(x, dict) for x in instance["ans"][key]
            )
            for event in instance["ans"][key]:
                assert event["event_type"] in instance["schema"]["event"]
                assert (
                    isinstance(event["trigger"], dict)
                    and event["trigger"]["text"] in instance["text"]
                    and instance["text"][
                        event["trigger"]["span"][0] : event["trigger"]["span"][1]
                    ]
                    == event["trigger"]["text"]
                )
                for arg in event["args"]:
                    assert (
                        arg["role"] in instance["schema"]["event"][event["event_type"]]
                    )
                    assert (
                        isinstance(arg["text"], str)
                        and instance["text"][arg["span"][0] : arg["span"][1]]
                        == arg["text"]
                    )
        elif key == "span":
            assert isinstance(instance["ans"][key], list) and all(
                isinstance(x, dict) for x in instance["ans"][key]
            )
            for span in instance["ans"][key]:
                assert (
                    isinstance(span["text"], str)
                    and instance["text"][span["span"][0] : span["span"][1]]
                    == span["text"]
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


def main():
    filepaths = []
    for filepath in filepaths:
        data = load_jsonlines(filepath)
        data_ok = True
        for ins in data:
            ok = is_valid_udi_instance(ins)
            if not ok:
                data_ok = False
                break
        if not data_ok:
            print(filepath)


if __name__ == "__main__":
    main()

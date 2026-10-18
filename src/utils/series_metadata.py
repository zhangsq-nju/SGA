import json
import os


METADATA_FILENAME = "series_metadata.jsonl"


def build_series_metadata(data):
    freq = data.get("freq", "D")
    seasonal_errors = data.get("seasonal_errors", [None] * len(data["ids"]))
    return [
        {
            "id": series_id,
            "freq": freq,
            "seasonal_error": seasonal_error,
        }
        for series_id, seasonal_error in zip(data["ids"], seasonal_errors)
    ]


def write_series_metadata(output_dir, data):
    metadata = build_series_metadata(data)
    path = os.path.join(output_dir, METADATA_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return metadata


def load_series_metadata(output_dir):
    path = os.path.join(output_dir, METADATA_FILENAME)
    if not os.path.isfile(path):
        return {}

    metadata = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            metadata[item["id"]] = item
            metadata[str(item["id"])] = item
    return metadata


def merge_series_metadata(items, metadata):
    merged = []
    for item in items:
        meta = metadata.get(item["id"], metadata.get(str(item["id"]), {}))
        if meta:
            merged.append(item | {k: v for k, v in meta.items() if k != "id"})
        else:
            merged.append(item)
    return merged

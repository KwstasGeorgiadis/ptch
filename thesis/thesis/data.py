import json
import random
from pathlib import Path

ARC_DIR = Path("/Users/kostas/dev/ARC-AGI/data/training")
REARC_DIR = Path("/Users/kostas/dev/re_arc/tasks")


def index_task_files(arc_dir=ARC_DIR, rearc_dir=REARC_DIR):
    index = {}

    for folder in (Path(arc_dir), Path(rearc_dir)):
        for path in sorted(folder.glob("*.json")):
            index.setdefault(path.stem, []).append(path)

    return index


def load_task_pairs(paths):
    pairs = []
    seen = set()

    for path in paths:
        data = json.loads(path.read_text())
        raw_pairs = data["train"] if isinstance(data, dict) else data

        for pair in raw_pairs:
            key = (
                tuple(map(tuple, pair["input"])),
                tuple(map(tuple, pair["output"])),
            )

            if key not in seen:
                seen.add(key)
                pairs.append({
                    "input": pair["input"],
                    "output": pair["output"],
                })

    return pairs


def split_tasks(index, seed=0, validation_fraction=0.2):
    task_ids = sorted(index)
    random.Random(seed).shuffle(task_ids)

    count = round(len(task_ids) * validation_fraction)

    training_ids = task_ids[count:]
    validation_ids = task_ids[:count]

    return training_ids, validation_ids


def sample_episode(index, task_ids, rng):
    task_id = rng.choice(task_ids)
    pairs = load_task_pairs(index[task_id])

    query = rng.choice(pairs)

    # Keep the query input out of the support examples.
    candidates = [
        pair for pair in pairs
        if pair["input"] != query["input"]
    ]

    support = rng.sample(candidates, min(3, len(candidates)))

    return task_id, support, query
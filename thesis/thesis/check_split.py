from .data import index_task_files, split_tasks


def check_split(index, train_ids, val_ids):
    assert not set(train_ids) & set(val_ids), "Task IDs overlap."

    assert (
        set(train_ids) | set(val_ids) == set(index)
    ), "Some tasks are missing."

    assert all(
        path.stem == task_id
        for task_id, paths in index.items()
        for path in paths
    ), "A file is grouped under a different task ID."

    training_tasks = {
        path.stem
        for task_id in train_ids
        for path in index[task_id]
    }

    validation_tasks = {
        path.stem
        for task_id in val_ids
        for path in index[task_id]
    }

    shared_tasks = training_tasks & validation_tasks

    assert not shared_tasks, "Tasks appear in both sets."

    print("Training tasks:", len(training_tasks))
    print("Validation tasks:", len(validation_tasks))
    print("Tasks appearing in both:", len(shared_tasks))


def main():
    index = index_task_files()
    train_ids, val_ids = split_tasks(index, seed=0)

    check_split(index, train_ids, val_ids)


if __name__ == "__main__":
    main()
def assign_data_to_tasks(data, targets_order, tasks_nr):
    data_grouped_by_tasks = [[] for _ in range(tasks_nr)]
    targets_nr = len(targets_order)
    targets_per_task = targets_nr // tasks_nr

    for (feature, target) in data:
        target_index_in_targets_orders = targets_order.index(target)
        task_index = target_index_in_targets_orders // targets_per_task

        if task_index >= tasks_nr:
            data_grouped_by_tasks[-1].append([feature, target])
        else:
            data_grouped_by_tasks[task_index].append([feature, target])

    return data_grouped_by_tasks

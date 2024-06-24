from icarl.config import visualize_exemplar_selection


def assign_data_to_tasks(data, targets_order, tasks_nr):
    max_nr_of_samples_per_target = float('inf')
    nr_of_samples_per_target = None
    data_grouped_by_tasks = [[] for _ in range(tasks_nr)]
    targets_nr = len(targets_order)
    targets_per_task = targets_nr // tasks_nr

    if visualize_exemplar_selection:
        # In order to have a meaningful visualization of the exemplar selection, the number of samples per target should
        # be reduced. We have chosen a maximum of 100 samples per target but this can be adjusted.
        max_nr_of_samples_per_target = 100
        nr_of_samples_per_target = {key: 0 for key in range(targets_nr)}

    for (feature, target) in data:
        if (
                target < targets_nr
                and (
                    not visualize_exemplar_selection
                    or nr_of_samples_per_target[target] < max_nr_of_samples_per_target
                )
        ):
            if visualize_exemplar_selection:
                nr_of_samples_per_target[target] += 1

            target_index_in_targets_orders = targets_order.index(target)
            task_index = target_index_in_targets_orders // targets_per_task

            if task_index >= tasks_nr:
                data_grouped_by_tasks[-1].append([feature, target])
            else:
                data_grouped_by_tasks[task_index].append([feature, target])

    return data_grouped_by_tasks

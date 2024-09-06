from time import time
from torch import manual_seed, device as imported_device, backends
from torch.nn import BCELoss
from .config import learning_rate_division_value, tasks_nr, targets_order, seed
from .ResNet32iCaRL import ResNet32iCaRL
from .train_one_incremental_learning_step import train_one_incremental_learning_step


def icarl(data_grouped_by_tasks, selection_method, log):
    manual_seed(seed)

    device = imported_device('mps' if backends.mps.is_available() else 'cpu')
    targets_nr = len(targets_order)
    model = ResNet32iCaRL(targets_nr)
    model = model.to(device)
    old_model = None
    loss_fn = BCELoss()
    gamma = 1.0 / learning_rate_division_value
    train_data_grouped_by_tasks, test_data_grouped_by_tasks = data_grouped_by_tasks
    current_test_data = []
    exemplars = []
    target_means_or_medians = {}
    accuracy_scores = []
    task_times = []

    for task_nr in range(tasks_nr):
        task_start_time = time()
        current_train_data = train_data_grouped_by_tasks[task_nr] + exemplars
        current_test_data += test_data_grouped_by_tasks[task_nr]
        old_model, exemplars = train_one_incremental_learning_step(model, old_model, device, loss_fn, task_nr,
                                                                   targets_nr, gamma, current_train_data,
                                                                   current_test_data, exemplars,
                                                                   target_means_or_medians, selection_method,
                                                                   accuracy_scores, log)
        task_end_time = time()
        task_execution_time = task_end_time - task_start_time

        log(f"Task execution time: {task_execution_time:.4f}s")

        task_times.append(task_execution_time)

    total_execution_time = sum(task_times)

    log(f"Total execution time: {total_execution_time:.4f}s")
    log(f"Accuracy scores: {accuracy_scores}")

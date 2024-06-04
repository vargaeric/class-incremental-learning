from torch import manual_seed, device as imported_device, backends
from torch.nn import BCELoss
from .config import learning_rate_division_value, tasks_nr, targets_order, seed
from .ResNet32 import ResNet32
from .train_one_incremental_learning_step import train_one_incremental_learning_step


def icarl(data_grouped_by_tasks, selection_method):
    manual_seed(seed)

    device = imported_device('mps' if backends.mps.is_available() else 'cpu')
    targets_nr = len(targets_order)
    model = ResNet32(targets_nr)
    model = model.to(device)
    old_model = None
    loss_fn = BCELoss()
    gamma = 1.0 / learning_rate_division_value

    train_data_grouped_by_tasks, test_data_grouped_by_tasks = data_grouped_by_tasks
    current_test_data = []
    exemplars = []
    train_data_grouped_by_targets = {}
    target_means_or_medians = {}

    for task_nr in range(tasks_nr):
        old_model, exemplars = train_one_incremental_learning_step(model, old_model, device, loss_fn, task_nr,
                                                                   targets_nr, gamma, train_data_grouped_by_tasks,
                                                                   test_data_grouped_by_tasks, current_test_data,
                                                                   exemplars, train_data_grouped_by_targets,
                                                                   target_means_or_medians, selection_method)

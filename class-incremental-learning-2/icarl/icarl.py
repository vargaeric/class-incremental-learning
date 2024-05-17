import torch
from torch.nn import BCELoss

from .config import learning_rate_division_value, tasks_nr, classes_order, seed
from .ResNet32 import ResNet32
from .train_one_incremental_learning_step import train_one_incremental_learning_step

def icarl(grouped_data_by_tasks, selection_method):
    torch.manual_seed(seed)

    classes_nr = len(classes_order)
    model = ResNet32(classes_nr)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    old_model = None
    loss_fn = BCELoss()
    gamma = 1.0 / learning_rate_division_value

    train_data_per_tasks, test_data_per_tasks = grouped_data_by_tasks
    current_test_data = []
    exemplars = []
    training_data_grouped_by_classes = {}
    class_means = {}

    for task_nr in range(tasks_nr):
        old_model, exemplars = train_one_incremental_learning_step(device, model, old_model, loss_fn, task_nr,
                                                                   classes_nr, gamma, train_data_per_tasks,
                                                                   test_data_per_tasks, current_test_data, exemplars,
                                                                   training_data_grouped_by_classes, class_means,
                                                                   selection_method)

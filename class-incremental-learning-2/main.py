from argparse import ArgumentParser
from torchvision.datasets import CIFAR100 as CIFAR100_dataset

from constants import ICARL, CIFAR100, ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION
from icarl.icarl import icarl
from icarl.config import (
    exemplars_nr_per_class, epochs_nr, learning_rate_starting_value, learning_rate_division_value,
    learning_rate_milestones, batch_size, weight_decay, momentum, tasks_nr, classes_order, seed
)

from data_handler import group_data_by_tasks


def main(method, dataset, selection):
    print(f"Training {method} with the following settings:")
    print(f" - Number of exemplars per class: {exemplars_nr_per_class}")
    print(f" - Number of epochs: {epochs_nr}")
    print(f" - Starting learning rate: {learning_rate_starting_value}")
    print(f" - Learning rate division: {learning_rate_division_value}")
    print(f" - Milestones for learning rate adjustment: {learning_rate_milestones}")
    print(f" - Size of the batches: {batch_size}")
    print(f" - Weight decay: {weight_decay}")
    print(f" - Momentum: {momentum}")
    print(f" - Number of tasks: {tasks_nr}")
    print(f" - Class order: {classes_order}")
    print(f" - Random seed: {seed}")

    data = None

    if dataset == CIFAR100:
        data = CIFAR100_dataset

    grouped_data_by_tasks = group_data_by_tasks(data, tasks_nr, classes_order)

    if method == ICARL:
        icarl(grouped_data_by_tasks, selection)


# To start the code run "python3 ./main.py --method=iCaRL --dataset=CIFAR100 --selection=Original" in the terminal!
if __name__ == "__main__":
    parser = ArgumentParser(description="Do incremental learning with the specified method.")
    parser.add_argument('--method', type=str, help='Incremental learning method', required=True,
                        choices=[ICARL])
    parser.add_argument('--dataset', type=str, help='Identifier of the dataset', required=True,
                        choices=[CIFAR100])
    parser.add_argument('--selection', type=str, help='Identifier of the exemplar selection method',
                        required=True, choices=[ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION,
                                                DENSITY_SELECTION])
    args = parser.parse_args()

    main(**vars(args))

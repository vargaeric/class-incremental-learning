from argparse import ArgumentParser
from torchvision.datasets import CIFAR100
from constants import (
    ICARL_APPROACH, CIFAR100_DATASET, ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION
)
from icarl.config import (
    exemplars_nr_per_target, epochs_nr, learning_rate_starting_value, learning_rate_division_value,
    learning_rate_milestones, batch_size, weight_decay, momentum, tasks_nr, targets_order, seed
)
from icarl.icarl import icarl
from data_handler.group_data_by_tasks import group_data_by_tasks


def main(approach, dataset, selection):
    print(f"Training {approach} with the following settings:")
    print(f" - Selection criteria: {selection}")
    print(f" - Dataset: {dataset}")
    print(f" - Number of exemplars per target: {exemplars_nr_per_target}")
    print(f" - Number of epochs: {epochs_nr}")
    print(f" - Starting learning rate: {learning_rate_starting_value}")
    print(f" - Learning rate division: {learning_rate_division_value}")
    print(f" - Milestones for learning rate adjustment: {learning_rate_milestones}")
    print(f" - Size of the batches: {batch_size}")
    print(f" - Weight decay: {weight_decay}")
    print(f" - Momentum: {momentum}")
    print(f" - Number of tasks: {tasks_nr}")
    print(f" - Targets order: {targets_order}")
    print(f" - Random seed: {seed}")

    data = None

    if dataset == CIFAR100_DATASET:
        data = CIFAR100

    data_grouped_by_tasks = group_data_by_tasks(data, tasks_nr, targets_order)

    if approach == ICARL_APPROACH:
        icarl(data_grouped_by_tasks, selection)


# To start the code run "python3 ./main.py --approach=iCaRL --dataset=CIFAR100 --selection=Original" in the terminal!
if __name__ == "__main__":
    parser = ArgumentParser(
        description="Do incremental learning with the specified approach, dataset and selection criterion."
    )
    parser.add_argument('--approach', type=str, help='Incremental learning approach', required=True,
                        choices=[ICARL_APPROACH])
    parser.add_argument('--dataset', type=str, help='Identifier of the dataset', required=True,
                        choices=[CIFAR100_DATASET])
    parser.add_argument('--selection', type=str, help='Identifier of the exemplar selection method', required=True,
                        choices=[ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION])

    args = parser.parse_args()

    main(**vars(args))

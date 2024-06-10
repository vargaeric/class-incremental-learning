from argparse import ArgumentParser
from os import makedirs, path
from functools import partial
from torchvision.datasets import CIFAR100
from constants import (
    ICARL_APPROACH, CIFAR100_DATASET, ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION
)
from icarl.config import (
    exemplars_nr_per_target, epochs_nr, learning_rate_starting_value, learning_rate_division_value,
    learning_rate_milestones, batch_size, weight_decay, momentum, tasks_nr, targets_order, seed
)
from icarl.icarl import icarl
from utils.get_current_date_and_time import get_current_date_and_time
from utils.print_and_log import print_and_log
from data_handlers.group_data_by_tasks import group_data_by_tasks


def main(approach, dataset, selection):
    results_folder_name = 'results'
    file_path = f'{results_folder_name}/{dataset}/{seed}/{tasks_nr}/{selection}-{get_current_date_and_time()}.txt'

    makedirs(path.dirname(file_path), exist_ok=True)

    file = open(file_path, 'x')
    log = partial(print_and_log, file)

    log(f"Training {approach} with the following settings:")
    log(f" - Dataset: {dataset}")
    log(f" - Selection criterion: {selection}")
    log(f" - Number of exemplars per target: {exemplars_nr_per_target}")
    log(f" - Number of epochs: {epochs_nr}")
    log(f" - Starting learning rate: {learning_rate_starting_value}")
    log(f" - Learning rate division: {learning_rate_division_value}")
    log(f" - Milestones for learning rate adjustment: {learning_rate_milestones}")
    log(f" - Size of the batches: {batch_size}")
    log(f" - Weight decay: {weight_decay}")
    log(f" - Momentum: {momentum}")
    log(f" - Number of tasks: {tasks_nr}")
    log(f" - Targets order: {targets_order}")
    log(f" - Random seed: {seed}")
    log()

    data = None

    if dataset == CIFAR100_DATASET:
        data = CIFAR100

    data_grouped_by_tasks = group_data_by_tasks(data, tasks_nr, targets_order)

    if approach == ICARL_APPROACH:
        icarl(data_grouped_by_tasks, selection, log)

    file.close()


# To start the code run "python3.8 ./main.py --approach=iCaRL --dataset=CIFAR100 --selection=Original" in the terminal!
if __name__ == "__main__":
    parser = ArgumentParser(
        description="Do class-incremental learning with the specified approach, dataset and selection criterion."
    )

    parser.add_argument('--approach', type=str, help="Identifier of the class-incremental learning approach",
                        required=True, choices=[ICARL_APPROACH])
    parser.add_argument('--dataset', type=str, help="Identifier of the dataset", required=True,
                        choices=[CIFAR100_DATASET])
    parser.add_argument('--selection', type=str, help="Identifier of the exemplar selection criterion method",
                        required=True,
                        choices=[ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION])

    args = parser.parse_args()

    main(**vars(args))

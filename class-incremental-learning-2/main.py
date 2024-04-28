import argparse
import json
from torchvision.datasets import CIFAR100
from incremental_learning_methods.icarl import icarl
from data_handler import group_data_by_tasks


def read_config(path):
    with open(path, 'r') as file:
        return json.load(file)


def main(method, dataset):
    config_path = f"./configs/{method.lower()}.json"
    config = read_config(config_path)

    print(f"Training {method} with the following settings:")
    print(f" - Number of exemplars per class: {config['exemplars_nr_per_class']}")
    print(f" - Number of epochs: {config['epochs_nr']}")
    print(f" - Starting learning rate: {config['learning_rate_starting_value']}")
    print(f" - Learning rate division: {config['learning_rate_division_value']}")
    print(f" - Milestones for learning rate adjustment: {config['learning_rate_milestones']}")
    print(f" - Size of the batches: {config['batch_size']}")
    print(f" - Weight decay: {config['weight_decay']}")
    print(f" - Momentum: {config['momentum']}")
    print(f" - Number of tasks: {config['tasks_nr']}")
    print(f" - Class order: {config['classes_order']}")
    print(f" - Random seed: {config['seed']}")

    data = None

    if dataset == 'CIFAR100':
        data = CIFAR100

    config['grouped_data_by_tasks'] = group_data_by_tasks(data, config['tasks_nr'], config['classes_order'])

    if method == 'iCaRL':
        icarl(**config)


# To start the code run "python3 ./main.py --method=iCaRL --dataset=CIFAR100" in the terminal!
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do incremental learning with the specified method.")
    parser.add_argument('--method', type=str, help='Incremental learning method', required=True,
                        choices=['iCaRL'])
    parser.add_argument('--dataset', type=str, help='Identifier of the dataset', required=True,
                        choices=['CIFAR100'])
    args = parser.parse_args()

    main(**vars(args))

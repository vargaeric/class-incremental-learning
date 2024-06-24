from os import path, makedirs
from utils.get_current_date_and_time import get_current_date_and_time


def save_exemplar_selection_visualization(plt, selection_method, target):
    visualization_folder_name = 'visualization'
    exemplar_selection_folder_name = 'exemplar-selection'
    file_name = f'{target}-{get_current_date_and_time()}'
    file_path = path.join(visualization_folder_name, exemplar_selection_folder_name, selection_method, file_name)

    makedirs(path.dirname(file_path), exist_ok=True)
    plt.savefig(f'{file_path}.png', dpi=200)

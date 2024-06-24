from matplotlib import pyplot as plt
from constants import DENSITY_SELECTION
from .save_exemplar_selection_visualization import save_exemplar_selection_visualization


def visualize_exemplar_selection_by_mean_or_median_or_density(returned_features_normalized, selected_exemplars,
                                                              mean_or_median, target, selection_method):
    plt.figure(figsize=(8, 6))
    plt.scatter(returned_features_normalized[:, 0], returned_features_normalized[:, 1], alpha=0.7)
    plt.scatter(selected_exemplars[:, 0], selected_exemplars[:, 1], color='C0', marker='s', s=100, edgecolors='black',
                linewidth=1)

    if not selection_method == DENSITY_SELECTION:
        plt.scatter(mean_or_median[0], mean_or_median[1], color='C0', marker='*', s=100, edgecolors='black',
                    linewidth=1)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    save_exemplar_selection_visualization(plt, selection_method, target)

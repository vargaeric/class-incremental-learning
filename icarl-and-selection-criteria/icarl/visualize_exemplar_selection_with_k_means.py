from matplotlib import pyplot as plt
from numpy import unique, where
from constants import K_MEANS_SELECTION
from .save_exemplar_selection_visualization import save_exemplar_selection_visualization


def visualize_exemplar_selection_with_k_means(returned_features_normalized, selected_exemplar_indices, centers, labels,
                                              target):
    plt.figure(figsize=(8, 6))

    for index, center, label in zip(selected_exemplar_indices, centers, unique(labels)):
        cluster_indices = where(labels == label)[0]

        plt.scatter(returned_features_normalized[cluster_indices, 0], returned_features_normalized[cluster_indices, 1],
                    alpha=0.7)
        plt.scatter(returned_features_normalized[index, 0], returned_features_normalized[index, 1], color=f'C{label}',
                    marker='s', s=100, edgecolors='black', linewidth=1)
        plt.scatter(center[0], center[1], color=f'C{label}', marker='*', s=100, edgecolors='black', linewidth=1)

    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    save_exemplar_selection_visualization(plt, K_MEANS_SELECTION, target)

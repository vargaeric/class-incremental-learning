import torch

from constants import ORIGINAL_SELECTION, MEDIAN_SELECTION, K_MEANS_SELECTION, DENSITY_SELECTION
from icarl.config import exemplars_nr_per_class, tasks_nr, classes_order
from .select_exemplars_with_kmeans import select_exemplars_with_kmeans
from .select_exemplars_by_density import select_exemplars_by_density


def define_class_means(model, device, task_nr, current_original_train_data, training_data_grouped_by_classes,
                       class_means, selection_method):
    classes_nr_in_one_task = int(len(classes_order) / tasks_nr)
    current_task_classes = classes_order[task_nr * classes_nr_in_one_task:(task_nr + 1) * classes_nr_in_one_task]

    selected_exemplars = []

    for current_task_class in current_task_classes:
        print(f"Selection exemplars for {current_task_class}...")

        original_data = training_data_grouped_by_classes[current_task_class]
        features = [item[0] for item in training_data_grouped_by_classes[current_task_class]]
        features_tensor = torch.stack(features)

        extracted_features_from_last_layer_2 = model(features_tensor.to(device) , return_features=True).detach().cpu()
        # print('-------------------')
        # print('extracted_features_from_last_layer')
        # print(np.asarray(extracted_features_from_last_layer_2).shape)
        # print('---------')
        # print(extracted_features_from_last_layer_2[0])
        # print('-------------------')
        D = extracted_features_from_last_layer_2.T
        D = D / torch.norm(D, dim=0)

        if selection_method == MEDIAN_SELECTION:
            mu = torch.median(D, dim=1)[0]
        else:
            mu = torch.mean(D, dim=1)

        # Storing or using the index
        class_means[current_task_class] = mu  # Store the mean

        if selection_method == ORIGINAL_SELECTION or selection_method == MEDIAN_SELECTION:
            D_transposed = D.T  # Correct the orientation
            similarity = torch.mv(D_transposed, mu)  # Corrected Matrix-vector multiplication

            top_k_values, top_k_indices = torch.topk(similarity, exemplars_nr_per_class)

            selected_exemplars_for_class = [original_data[idx] for idx in top_k_indices.tolist()]

            selected_exemplars += selected_exemplars_for_class

    if selection_method == K_MEANS_SELECTION:
        selected_exemplars = select_exemplars_with_kmeans(model, device, current_original_train_data, exemplars_nr_per_class, class_means)
    elif selection_method == DENSITY_SELECTION:
        selected_exemplars = select_exemplars_by_density(model, device, current_original_train_data, exemplars_nr_per_class)

    return selected_exemplars

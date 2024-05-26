def group_training_data_by_classes(training_data_grouped_by_classes, current_original_train_data):
    for current_original_train_data in current_original_train_data:
        feature, target = current_original_train_data

        if target in training_data_grouped_by_classes:
            training_data_grouped_by_classes[target].append([feature, target])
        else:
            training_data_grouped_by_classes[target] = [[feature, target]]

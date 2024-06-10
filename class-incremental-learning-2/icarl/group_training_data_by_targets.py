def group_training_data_by_targets(train_data_grouped_by_targets, current_original_train_data):
    for feature, target in current_original_train_data:
        if target in train_data_grouped_by_targets:
            train_data_grouped_by_targets[target].append(feature)
        else:
            train_data_grouped_by_targets[target] = [feature]

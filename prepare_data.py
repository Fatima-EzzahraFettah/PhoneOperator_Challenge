def construct_data(features, labels, target_name):
    data = features.copy()
    data[target_name] = labels
    return data
import numpy as np
from decision_tree import build_tree, predict
from collections import Counter
from decision_tree import predict


def bagging_predict(trees, row):
    from decision_tree import predict
    from collections import Counter

    predictions = [predict(tree, row)
                   for tree in trees]  # row must be a list of features
    return Counter(predictions).most_common(1)[0][0]


def subsample(dataset, ratio):
    n_sample = round(len(dataset) * ratio)
    indices = np.random.choice(len(dataset), n_sample, replace=True)
    return dataset[indices]


# random_forest.py
def random_forest(train, test, max_depth, min_size, sample_size, n_trees):
    trees = []
    for _ in range(n_trees):
        sample = subsample(train, sample_size)
        tree = build_tree(sample, max_depth, min_size)
        trees.append(tree)
    predictions = [bagging_predict(trees, row) for row in test]
    return trees  # Return trees instead of predictions

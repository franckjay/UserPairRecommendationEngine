import random
import logging
from src.utils.users import user_scores
from utils.elwc import construct_map, elwc_to_tfrecords_writer


def build_ranking_sets(train_fname, valid_fname, idx_split=9, num_pairs=3) -> None:
    """
    Super hacky way of getting unique rankings of user-item lists
    :param train_fname: Filename to write the training data to
    :param valid_fname: Same as above, valid data
    :param idx_split: The indices of items are ordered in values of when they were scored. Split on time to prevent
        leakage.
    :param num_pairs: How many items are included in every elwc
    :return:
    """

    training = []
    valid = []
    for v in range(1000):
        # Our train test split happens after 9 trials, the rest are validation
        iteration = sorted(
            random.choices(
                [idx for idx, v in user_scores["1"].items() if idx <= idx_split],
                k=num_pairs,
            )
        )
        if iteration not in training and len(set(iteration)) == 3:
            training.append(iteration)
        iteration = sorted(
            random.choices(
                [idx for idx, v in user_scores["1"].items() if idx > idx_split],
                k=num_pairs,
            )
        )
        if iteration not in valid and len(set(iteration)) == 3:
            valid.append(iteration)
    logging.info("Constructing training ELWC")
    training_elwc = construct_map(training)
    elwc_to_tfrecords_writer(training_elwc, train_fname)
    logging.info(
        "Finished building+writing %s Training records. Beginning valid:",
        str(len(training_elwc)),
    )
    valid_elwc = construct_map(valid)
    elwc_to_tfrecords_writer(valid_elwc, valid_fname)
    logging.info(
        "Finished building+writing %s validation records.", str(len(valid_elwc))
    )
    return

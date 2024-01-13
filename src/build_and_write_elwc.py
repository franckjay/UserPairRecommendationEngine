import random
from src.utils.users import user_scores
from utils.elwc import construct_map


training = []
valid = []
for v in range(1000):
    # Our train test split happens after 9 trials, the rest are validation
    iteration = sorted(
        random.choices([k for k, v in user_scores["1"].items() if k <= 9], k=3)
    )
    if iteration not in training and len(set(iteration)) == 3:
        training.append(iteration)
    iteration = sorted(
        random.choices([k for k, v in user_scores["1"].items() if k > 9], k=3)
    )
    if iteration not in valid and len(set(iteration)) == 3:
        valid.append(iteration)

training_elwc = construct_map(training)
valid_elwc = construct_map(valid)
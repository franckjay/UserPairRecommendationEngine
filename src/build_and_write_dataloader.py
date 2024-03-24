import random
import logging
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
from sklearn.preprocessing import RobustScaler

from utils.users import user_age, user_scores
from utils.embeddings import build_embeddings
from utils.products import product_score, product_description, product_name


def construct_map(output):
    example_list = []
    idx = 0
    for items in output:
        user_1, user_2 = (1, 2) if random.random() > 0.5 else (2, 1)

        context = {
            "user_id_1": int(user_1),
            "user_id_2": int(user_2),
            "user_age_1": float(user_age.get(user_1, None)),
            "user_age_2": float(user_age.get(user_2, None)),
        }
        examples = []
        for item in items:
            # We need to grab the user, the item, and the combined score
            base_dict = {
                "combined_score": float(
                    user_scores["1"][item] + user_scores["2"][item]
                ),
                "product_score": float(product_score[item]),
                "product_id": int(item),
                "product_description_embedding": build_embeddings(
                    product_description[item]
                ),
                "product_name_embedding": build_embeddings(product_name[item]),
            }

            examples.append(base_dict)
        example_list.append((context, examples))
        idx += 1
        if idx % 200 == 0:
            print("Finished writing idx ", idx)
            print(base_dict["product_id"], base_dict["product_score"])
    return example_list


def build_pandas_ranking(data_map):
    # Initialize empty list to store all data per row
    all_data = defaultdict(lambda: [])
    # Loop through training_map
    for i, (con_features, item_list) in enumerate(data_map):
        for feat, val in con_features.items():
            all_data[feat].append(val)
        # Extract document data
        target_var = []
        for j, doc in enumerate(item_list):
            for feat, val in doc.items():
                if "embedding" in feat:
                    for idx, flt in enumerate(val):
                        all_data[feat + f"_{j}_{idx}"].append(flt)
                else:
                    all_data[feat + f"_{j}"].append(val)

                if "combined_score" in feat:
                    # We are going to get all of the combined scores
                    target_var.append(val)
            # The target 'class' is the greatest of the scores
        all_data["target"].append(torch.argmax(torch.Tensor(target_var)).item())
    # Convert to DataFrame
    return pd.DataFrame(all_data)


class DictDataset(Dataset):
    def __init__(self, data_dict, norm_target=1, scaler=None):
        self.norm_target = norm_target
        self.data_df = build_pandas_ranking(data_dict)
        self.scaler = scaler

        # Build out the features that are continuous variables
        self.float_features = []

        for feat in self.data_df.columns:
            valid = True
            # TODO ensure no targets/ combined scores, or ids are in the data!
            for non_float_keyword in [
                "user_id",
                "product_id",
                "combined_score",
                "target",
            ]:
                if non_float_keyword in feat:
                    valid = False
                    continue
            if valid:
                self.float_features.append(feat)

        self.u_cats = [
            _feat for _feat in self.data_df.columns if _feat.startswith("user_id")
        ]
        self.i_cats = [
            _feat for _feat in self.data_df.columns if _feat.startswith("product_id")
        ]
        self.targets = [
            _feat
            for _feat in self.data_df.columns
            if _feat.startswith("combined_score")
        ]

        logging.debug(
            "Float features: %s",
            [_ for _ in self.float_features if "embedding" not in _],
        )
        logging.debug("User Embedding features: %s", self.u_cats)
        logging.debug("Item Categorical features: %s", self.i_cats)
        logging.debug("Target features: %s", self.targets)

        if not self.scaler:
            # If we haven't trained a scaler yet, do so here
            self.scaler = RobustScaler().fit(self.data_df[self.float_features])
        # Scale the data from the float features into its own dataframe
        self.float_df = pd.DataFrame(
            self.scaler.transform(self.data_df[self.float_features]),
            columns=self.float_features,
        )
        # Drop these from the normal DF as they are un-scaled
        self.data_df = self.data_df.drop(self.float_features, axis=1)

    def __getitem__(self, index):

        return (
            torch.Tensor(self.float_df.iloc[index].values),
            torch.Tensor(self.data_df[self.u_cats].iloc[index].values),
            torch.Tensor(self.data_df[self.i_cats].iloc[index].values),
            torch.Tensor(
                self.data_df[self.targets].iloc[index].values / self.norm_target
            ),
        )

    def __len__(self):
        return len(self.data_df)


def build_train_valid_loaders(n_items_ranked=3, batch=5):
    """
    Builds the DataLoaders for train and validation sets
    :param n_items_ranked: How many items do  you want ranked?
    :param batch: What is our batch size
    :return: DataLoaders for train/valid
    """
    training = []
    valid = []
    # Just a dumb way of building out all the permutations
    for v in range(1000):
        # Our train test split happens after 12 trials, the rest are validation
        iteration = random.choices(
            [k for k, v in user_scores["1"].items() if k <= 12], k=n_items_ranked
        )
        if iteration not in training and len(set(iteration)) == 3:
            training.append(iteration)
        iteration = random.choices(
            [k for k, v in user_scores["1"].items() if k > 12], k=n_items_ranked
        )
        if iteration not in valid and len(set(iteration)) == 3:
            valid.append(iteration)

    training_map = construct_map(training)
    valid_map = construct_map(valid)
    # Build the train_ds, and use that scaler for the validation set
    train_ds = DictDataset(training_map, norm_target=20)
    valid_ds = DictDataset(valid_map, norm_target=20, scaler=train_ds.scaler)
    train_loader = DataLoader(train_ds, batch_size=batch)
    valid_loader = DataLoader(valid_ds, batch_size=batch)
    return train_loader, valid_loader, train_ds, valid_ds

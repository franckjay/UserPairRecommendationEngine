import logging
import numpy as np
from pandas import DataFrame
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from build_model import EmbeddingRankingModel
from build_and_write_dataloader import DictDataset


def torch_top_1(actual: list[float], preds: list[float]) -> int:
    score = 0
    for a, p in zip(actual, preds):
        score += 1 if np.argmax(a) == np.argmax(p) else 0
    return score / len(actual)


def train_er_model(
    er: EmbeddingRankingModel,
    loss_fn,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    opt,
    n_epochs: int,
):
    lr_vals = []
    training_losses = []
    valid_losses = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    er = er.to(device)
    for epoch in range(n_epochs):
        y_true_train = list()
        y_pred_train = list()
        total_loss_train = 0

        for cont, u_cat, i_cat, y in train_loader:
            if device != "cpu":
              cont = cont.cuda()
              u_cat = u_cat.cuda()
              i_cat = i_cat.cuda()
              y = y.cuda()
            pred = er.forward(cont, u_cat, i_cat)
            loss = loss_fn(pred, y)

            loss.backward()  # do backprop
            lr_vals.append(opt.param_groups[0]["lr"])
            training_losses.append(loss.item())
            opt.step()

            y_true_train += list(y.cpu().data.numpy())
            y_pred_train += list(pred.cpu().data.numpy())
            total_loss_train += loss.item()

        train_score = ndcg_score(y_true_train, y_pred_train)
        train_loss = total_loss_train / len(
            train_loader
        )  # len train_dl = 704. the calc is number of train examples (89991) / batch size (128)
        opt.zero_grad()  # find where the grads are zero
        train_acc = torch_top_1(y_true_train, y_pred_train)
        if valid_loader:
            y_true_val = list()
            y_pred_val = list()
            total_loss_val = 0
            for cont, u_cat, i_cat, y in valid_loader:
                if device != "cpu":
                    cont = cont.cuda()
                    u_cat = u_cat.cuda()
                    i_cat = i_cat.cuda()
                    y = y.cuda()
                pred = er.forward(cont, u_cat, i_cat)
                loss = loss_fn(pred, y)

                y_true_val += list(y.cpu().data.numpy())
                y_pred_val += list(pred.cpu().data.numpy())
                total_loss_val += loss.item()
                valid_losses.append(loss.item())
            val_score = ndcg_score(y_true_val, y_pred_val)
            val_loss = total_loss_val / len(valid_loader)
            val_acc = torch_top_1(y_true_val, y_pred_train)
            logging.info(
                f"Epoch {epoch}: train_loss: {train_loss:.4f} train_ndcg: {train_score:.4f} train_acc_1: {train_acc:.4f} | val_loss: {val_loss:.4f} val_ndcg: {val_score:.4f} val_acc_1: {val_acc:.4f}"
            )
        else:
            logging.info(
                f"Epoch {epoch}: train_loss: {train_loss:.4f} train_ndcg: {train_acc:.4f}"
            )

    plt.plot(range(len(training_losses)), training_losses)
    plt.plot(range(len(valid_losses)), valid_losses)
    plt.show()


def train_sklearn_ranker(
    train_ds: DictDataset,
    valid_ds: DictDataset,
    train_targ: DataFrame,
    valid_targ: DataFrame,
) -> LogisticRegression:
    """
    Trains and returns a LogReg model.
    :param train_ds: Dataset of training data
    :param valid_ds: Validation Dataset
    :param train_targ: Target class we are trying to hit
    :param valid_targ: Validation class
    :return: LogReg model
    """

    feature_list = [
        _feat
        for _feat in train_ds.float_df.columns
        if (_feat != "target" and "combined" not in _feat)
    ]
    model = LogisticRegression(
        C=0.000001, multi_class="multinomial", max_iter=10000, n_jobs=5
    )
    model.fit(train_ds.float_df[feature_list], train_targ)
    logging.info(
        "Validation score: %s",
        str(model.score(valid_ds.float_df[feature_list], valid_targ)),
    )
    train_preds = model.predict_proba(train_ds.float_df[feature_list])
    valid_preds = model.predict_proba(valid_ds.float_df[feature_list])
    logging.info(
        f"NDCG Scores: Train - {ndcg_score(train_ds.data_df[train_ds.targets].values, train_preds)}"
    )
    logging.info(
        f"Validation - {ndcg_score(valid_ds.data_df[valid_ds.targets].values, valid_preds)}"
    )
    return model

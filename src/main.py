import logging
import torch

from build_and_write_dataloader import build_train_valid_loaders
from build_model import EmbeddingRankingModel
from model_trainers import train_sklearn_ranker, train_er_model


def main(
    batch_size: int, list_size: int, n_epochs: int, lr: float,
):
    logging.info("Building the DataLoaders for ranking")
    train_dl, valid_dl, train_ds, valid_ds = build_train_valid_loaders(
        list_size, batch_size
    )
    logging.info("Constructing the model")
    er = EmbeddingRankingModel(n_docs=list_size, batch_size=batch_size)

    logging.info("Training the Torch model")
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(er.parameters(), lr=lr, weight_decay=0.2)
    train_er_model(er, loss_fn, train_dl, valid_dl, opt, n_epochs)
    logging.info("Job done!")

    logging.info("Training the LogReg model")
    train_sklearn_ranker(
        train_ds, valid_ds, train_ds.data_df[["target"]], valid_ds.data_df[["target"]]
    )
    logging.info("Job done!")


if __name__ == "__main__":
    # TODO: run argparse here
    logging.info("Let's make a Coffee Tasting Model!")

    n_batch, n_list, epochs, learn_rate = 5, 3, 5, 0.05
    main(n_batch, n_list, epochs, learn_rate)

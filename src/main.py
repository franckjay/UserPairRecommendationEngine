import logging
from build_and_write_elwc import build_ranking_sets
from build_model_feature_spec import build_ranking_pipeline


def main(
    training_data_path: str,
    valid_data_path: str,
    batch_size: int,
    list_size: int,
    n_epochs: int,
    lr: float,
):
    logging.info("Building the ELWCs")
    build_ranking_sets()
    logging.info("Constructing the model")
    constructed_ranker = build_ranking_pipeline(
        [64, 32],
        training_data_path,
        valid_data_path,
        batch_size,
        list_size,
        n_epochs,
        lr,
    )
    logging.info("Training the model")
    constructed_ranker.train_and_validate(verbose=2)
    logging.info(
        "Job done! Model should be found in `/tmp/ranking_model_dir/export/latest_model`"
    )


if __name__ == "__main__":
    # TODO: run argparse here
    logging.info("Let's make a Coffee Tasting Model!")
    train_df_name, valid_df_name = "coffee_training.tfrecords", "coffee_valid.tfrecords"
    n_batch, n_list, epochs, learn_rate = 5, 3, 5, 0.05
    main(train_df_name, valid_df_name, n_batch, n_list, epochs, learn_rate)

import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr

# TODO: take these as input in main.py through argparse
N_USERS = 2
N_ITEMS = 30
USER_EMBED_SIZE = 10
TEXT_EMBED_SIZE = 768

# Why am I filling these user embeddings with random variables? https://stackoverflow.com/questions/58003010/tensorflow-tfrecords-cannot-parse-serialized-example
context_feature_spec = {
    "user_id_1": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
    "user_id_2": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
    "user_age_1": tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.float32, default_value=18.0
    ),
    "user_age_2": tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.float32, default_value=18.0
    ),
    "user_embedding_1": tf.io.FixedLenFeature(
        shape=(USER_EMBED_SIZE,),
        dtype=tf.float32,
        default_value=np.random.randn(USER_EMBED_SIZE),
    ),
    "user_embedding_2": tf.io.FixedLenFeature(
        shape=(USER_EMBED_SIZE,),
        dtype=tf.float32,
        default_value=np.random.randn(USER_EMBED_SIZE),
    ),
}
example_feature_spec = {
    "product_score": tf.io.FixedLenFeature(
        shape=(1,), dtype=tf.float32, default_value=0
    ),
    "product_id": tf.io.FixedLenFeature(shape=(1,), dtype=tf.int64, default_value=0),
    "product_description_embedding": tf.io.FixedLenFeature(
        shape=(TEXT_EMBED_SIZE,),
        dtype=tf.float32,
        default_value=np.random.randn(TEXT_EMBED_SIZE),
    ),
    "product_name_embedding": tf.io.FixedLenFeature(
        shape=(TEXT_EMBED_SIZE,),
        dtype=tf.float32,
        default_value=np.random.randn(TEXT_EMBED_SIZE),
    ),
    "product_embedding": tf.io.FixedLenFeature(
        shape=(TEXT_EMBED_SIZE,),
        dtype=tf.float32,
        default_value=np.random.randn(TEXT_EMBED_SIZE),
    ),
}
label_spec = (
    "combined_score",
    tf.io.FixedLenFeature(shape=(1,), dtype=tf.float32, default_value=-1),
)


class UserItemPreprocessor(tfr.keras.model.Preprocessor):
    def __init__(self, embedding_dim, n_users, n_items):
        self._embedding_dim = embedding_dim
        self.n_users = n_users
        self.n_items = n_items

    def __call__(self, context_inputs, example_inputs, mask):
        user_lookup = tf.keras.layers.StringLookup(mask_token=None)
        user_embedding = tf.keras.layers.Embedding(
            input_dim=self.n_users,
            output_dim=self._embedding_dim,
            embeddings_initializer=None,
            embeddings_constraint=None,
        )

        item_lookup = tf.keras.layers.StringLookup(mask_token=None)
        item_embedding = tf.keras.layers.Embedding(
            input_dim=self.n_items,
            output_dim=self._embedding_dim,
            embeddings_initializer=None,
            embeddings_constraint=None,
        )

        context_inputs["user_embedding_1"] = tf.reduce_mean(
            user_embedding(user_lookup(context_inputs["user_id_1"])), axis=-2
        )
        context_inputs["user_embedding_2"] = tf.reduce_mean(
            user_embedding(user_lookup(context_inputs["user_id_2"])), axis=-2
        )
        example_inputs["product_embedding"] = tf.reduce_mean(
            item_embedding(item_lookup(example_inputs["product_id"])), axis=-2
        )
        del context_inputs["user_id_1"]
        del context_inputs["user_id_2"]
        del example_inputs["product_id"]

        return context_inputs, example_inputs


def build_ranking_pipeline(
    layer_list: list[int],
    train_tf_path: str,
    valid_tf_path: str,
    batch_size: int,
    list_size: int,
    n_epochs: int,
    lr: float,
) -> tfr.keras.pipeline.SimplePipeline:
    scorer = tfr.keras.model.DNNScorer(
        hidden_layer_dims=layer_list,
        output_units=1,
        activation=tf.nn.relu,
        use_batch_norm=True,
    )
    # Using the CON/EX feature specs, create a combined version
    input_creator = tfr.keras.model.FeatureSpecInputCreator(
        context_feature_spec, example_feature_spec
    )
    # With the embeddings, we need to do some pre-processing per batch
    pp = UserItemPreprocessor(USER_EMBED_SIZE, N_USERS, N_ITEMS)
    #
    model_builder = tfr.keras.model.ModelBuilder(
        input_creator=input_creator,
        preprocessor=pp,
        scorer=scorer,
        mask_feature_name="example_list_mask",
        name="coffee_model",
    )

    # If you want to plot your model as a check, this can be useful
    # model = model_builder.build()
    # tf.keras.utils.plot_model(model, expand_nested=True)

    dataset_hparams = tfr.keras.pipeline.DatasetHparams(
        train_input_pattern=train_tf_path,
        valid_input_pattern=valid_tf_path,
        train_batch_size=batch_size,
        valid_batch_size=batch_size,
        list_size=list_size,
        dataset_reader=tf.data.TFRecordDataset,
    )

    dataset_builder = tfr.keras.pipeline.SimpleDatasetBuilder(
        context_feature_spec,
        example_feature_spec,
        mask_feature_name="example_list_mask",
        label_spec=label_spec,
        hparams=dataset_hparams,
    )

    pipeline_hparams = tfr.keras.pipeline.PipelineHparams(
        # TODO: Take model path as input
        model_dir="/tmp/ranking_model_dir",
        num_epochs=n_epochs,
        steps_per_epoch=1000,
        validation_steps=100,
        learning_rate=lr,
        loss="softmax_loss",
        strategy="MirroredStrategy",
    )

    ranking_pipeline = tfr.keras.pipeline.SimplePipeline(
        model_builder, dataset_builder=dataset_builder, hparams=pipeline_hparams
    )
    return ranking_pipeline

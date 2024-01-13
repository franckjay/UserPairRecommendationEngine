import random
import logging
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_serving.apis import input_pb2
from google.protobuf import text_format


from tensorflow_serving.apis import input_pb2
import tensorflow as tf


from embeddings import build_embeddings
from users import user_scores
from products import product_score, product_description, product_name


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(value)]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def to_example(dictionary):
    return tf.train.Example(features=tf.train.Features(feature=dictionary))


def construct_elwc(context: dict, examples: list[dict]):
    """
    Pass in a Context and Example set, construct an ELWC example.
    :param context: Context for each ranking
    :param examples: List of Dicts, each corresponding
    :return:
    """
    ELWC = input_pb2.ExampleListWithContext()
    ELWC.context.CopyFrom(to_example(context))
    for expl in examples:
        example_features = ELWC.examples.add()
        example_features.CopyFrom(to_example(expl))
    return ELWC


def elwc_to_tfrecords_writer(
    elwc_list: list[input_pb2.ExampleListWithContext], fname: str
):
    """
    I take in a list of elwc objects, a filename, and then dump out some TF Records
    :param elwc_list: Each of the examples for our train/test set
    :param fname: Filename you want to write ot
    :return:
    """
    logging.info("Writing out TFRecords")
    with tf.io.TFRecordWriter(fname) as writer:
        for elwc in elwc_list:
            writer.write(elwc.SerializeToString())
    writer.close()
    logging.info(
        "Finishing writing %s TFREcords to path %s", str(len(elwc_list)), str(fname)
    )
    return


def construct_map(output):
    elwc_list = []
    idx = 0
    for items in output:
        user_1, user_2 = (1, 2) if random.random() > 0.5 else (2, 1)

        context = {
            "user_id_1": _int64_feature(user_1),
            "user_id_2": _int64_feature(user_2),
            "user_age_1": _float_feature(user_age.get(user_1, None)),
            "user_age_2": _float_feature(user_age.get(user_2, None)),
        }
        examples = []
        for item in items:
            # We need to grab the user, the item, and the combined score
            base_dict = {
                "combined_score": _float_feature(
                    user_scores["1"][item] + user_scores["2"][item]
                ),
                "product_score": _float_feature(product_score[item]),
                "product_id": _int64_feature(item),
                "product_description_embedding": _list_feature(
                    build_embeddings(product_description[item])
                ),
                "product_name_embedding": _list_feature(
                    build_embeddings(product_name[item])
                ),
            }

            examples.append(base_dict)
        elwc_list.append(construct_elwc(context, examples))
        idx += 1
        if idx % 200 == 0:
            print("Finished writing idx ", idx)
            print(base_dict["product_id"], base_dict["product_score"])
    return elwc_list

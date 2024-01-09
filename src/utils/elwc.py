
import logging
import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow_serving.apis import input_pb2
from google.protobuf import text_format


from tensorflow_serving.apis import input_pb2
import tensorflow as tf

# https://stackoverflow.com/questions/62004669/tf-ranking-transform-data-to-elwc-examplelistwithcontext-form

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

def elwc_to_tfrecords_writer(elwc_list: list[input_pb2.ExampleListWithContext], fname: str):
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
    logging.info("Finishing writing %s TFREcords to path %s", str(len(elwc_list)), str(fname))
    return


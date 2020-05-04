import tensorflow as tf
import numpy as np
from preprocess.features import process as process_features
from preprocess.labels import process as process_labels

batch_size = 16


def tf_function_wrapper(features, labels):
    features, feature_length = tf.py_function(func=process_features, inp=[
        features['filename']],
        Tout=(tf.float32, tf.int32))
    labels, label_length = tf.py_function(func=process_labels, inp=[
        labels], Tout=(tf.int32, tf.int32))

    inputs = {'the_input': features,
              'the_labels': labels,
              'input_length': feature_length,
              'label_length': label_length}

    # dummy data for dummy loss function
    outputs = {'ctc': np.zeros([batch_size])}

    return inputs, outputs


def remove_big_sentences(x, y):
    comparison = tf.math.less(tf.strings.length(y), tf.constant(280))
    return tf.math.count_nonzero(comparison) == len(y)


def load_dataset(csv_filename):
    librivox_paths_ds = tf.data.experimental.make_csv_dataset(
        csv_filename, batch_size=batch_size,
        shuffle=False, label_name="transcript")

    librivox_paths_ds = librivox_paths_ds.filter(remove_big_sentences)
    librivox_ds = librivox_paths_ds.map(
        tf_function_wrapper, num_parallel_calls=6)
    librivox_ds = librivox_ds.prefetch(buffer_size=1)

    return librivox_ds

import tensorflow as tf
from preprocess.features import process as process_features
from preprocess.labels import process as process_labels


def tf_function_wrapper(features, labels):
    features, feature_lengths = tf.py_function(func=process_features, inp=[
        features['filename']],
        Tout=(tf.float32, tf.int32))
    labels, label_lengths = tf.py_function(func=process_labels, inp=[
        labels], Tout=(tf.int32, tf.int32))
    return features, labels


def load_dataset(csv_filename):
    librivox_paths_ds = tf.data.experimental.make_csv_dataset(
        csv_filename, batch_size=4,
        shuffle=False, label_name="transcript")

    librivox_ds = librivox_paths_ds.map(tf_function_wrapper)
    return librivox_ds

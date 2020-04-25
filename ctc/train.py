from preprocess.data import load_dataset
import tensorflow as tf

librivox_ds = load_dataset("librivox-test-clean-wav.csv")
for feature_batch, labels in librivox_ds.take(1):
    tf.print(feature_batch)

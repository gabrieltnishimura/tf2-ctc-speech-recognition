from preprocess.data import load_dataset
import tensorflow as tf

librivox_ds = load_dataset("librivox-test-clean-wav.csv")
for inputs, outputs in librivox_ds.take(1):
    tf.print(inputs)

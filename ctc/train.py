import tensorflow as tf
from preprocess.data import load_dataset
from models.brnn import brnn

librivox_ds = load_dataset("librivox-test-clean-wav.csv")
for inputs, outputs in librivox_ds.take(1):
    tf.print(inputs)


model = brnn(512)
loss = {'ctc': lambda y_true, y_pred: y_pred}
model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(
    lr=0.0001, epsilon=1e-8, clipnorm=2.0))
model.summary()

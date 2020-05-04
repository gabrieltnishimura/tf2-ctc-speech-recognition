import tensorflow as tf
from models.brnn_model import brnn_model

brnn = brnn_model(512, batch_size=4)
loss = {'ctc': lambda y_true, y_pred: y_pred}
optimizer = tf.keras.optimizers.Adam(
    lr=0.0001,
    epsilon=1e-8,
    clipnorm=2.0
)
brnn.compile(loss=loss, optimizer=optimizer)

__all__ = [
    'brnn',
]

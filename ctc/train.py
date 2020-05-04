import tensorflow as tf
from preprocess.data import load_dataset
from models import brnn
from training.LossCallback import LossCallback
from datetime import datetime

training_ds = load_dataset("librivox-train-clean-100-wav.csv")
validation_ds = load_dataset("librivox-dev-clean-wav.csv")
test_ds = load_dataset("librivox-test-clean-wav.csv")

# for features, labels in training_ds.take(1):
#     tf.print(tf.rank(features['the_input']))
#     tf.print(tf.rank(features['the_labels']))
#     tf.print(tf.rank(features['input_length']))
#     tf.print(tf.rank(features['label_length']))
#     tf.print(features['the_input'].get_shape())
#     tf.print(features['the_labels'].get_shape())
#     tf.print(features['input_length'].get_shape())
#     tf.print(features['label_length'].get_shape())

print("\n\nModel and training parameters: ")
print("Starting time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

tf.keras.utils.plot_model(brnn, 'my_first_model.png')
brnn.summary()

input_data = brnn.get_layer('the_input').input
y_pred = brnn.get_layer('ctc').input[0]
test_func = tf.keras.backend.function([input_data], [y_pred])

# The loss callback function that calculates WER while training
loss_cb = LossCallback(
    test_func=test_func,
    validation_gen=validation_ds.as_numpy_iterator(),
    test_gen=test_ds.as_numpy_iterator(),
    model=brnn,
    checkpoint=10,
    path_to_save='./',
    log_file_path='log'
)

brnn.fit(
    x=training_ds.as_numpy_iterator(),
    validation_data=validation_ds.as_numpy_iterator(),
    epochs=10,
    verbose=2,
    workers=1,
    shuffle=False,
    # callbacks=[loss_cb],
    use_multiprocessing=False,
)

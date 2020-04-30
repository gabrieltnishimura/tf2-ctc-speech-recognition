import tensorflow as tf
from preprocess.data import load_dataset
from models import brnn
from training.LossCallback import LossCallback
from datetime import datetime

training_ds = load_dataset("librivox-train-clean-100-wav.csv")
validation_ds = load_dataset("librivox-dev-clean-wav.csv")
test_ds = load_dataset("librivox-test-clean-wav.csv")

print("\n\nModel and training parameters: ")
print("Starting time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

brnn.summary()

input_data = brnn.get_layer('the_input').input
y_pred = brnn.get_layer('ctc').input[0]
test_func = tf.keras.backend.function([input_data], [y_pred])

# The loss callback function that calculates WER while training
loss_cb = LossCallback(
    test_func=test_func,
    validation_gen=validation_ds,
    test_gen=test_ds,
    model=brnn,
    checkpoint=10,
    path_to_save='./',
    log_file_path='log'
)

brnn.fit(
    x=training_ds,
    # validation_data=validation_ds,
    # epochs=10,
    # verbose=2,
    # workers=1,
    # shuffle=False,
    # callbacks=[loss_cb]
)

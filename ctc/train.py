import tensorflow as tf
from preprocess.data import load_dataset
from models import brnn
from training.LossCallback import LossCallback
from datetime import datetime
from config import ApplicationArguments

args = ApplicationArguments()
training_ds = load_dataset(args.trainDataset)
validation_ds = load_dataset(args.validationDataset)
test_ds = load_dataset(args.testDataset)

print("\n\nModel and training parameters: ")
print("Starting time: ", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

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
    shuffle=False,
    callbacks=[loss_cb],
)

import tensorflow as tf
from tensorflow.keras.callbacks import LossCallback

# Loss callback parameters
loss_callback_params = {
    'validation_gen': validation_generator,
    'test_gen': test_generator,
}

# Model training parameters
model_train_params = {
    'generator': training_generator,
    'epochs': epochs,
    'verbose': 2,
    'validation_data': validation_generator,
    'workers': 1,
    'shuffle': shuffle
}

callbacks = []
# Creates a test function that takes preprocessed sound input and outputs predictions
# Used to calculate WER while training the network
input_data = model.get_layer('the_input').input
y_pred = model.get_layer('ctc').input[0]
test_func = tf.keras.backend.function([input_data], [y_pred])

# The loss callback function that calculates WER while training
loss_cb = LossCallback(test_func=test_func,
                       model=model, **loss_callback_params)
callbacks.append(loss_cb)
